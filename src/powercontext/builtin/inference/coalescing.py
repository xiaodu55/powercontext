# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coalescing embedding wrapper with bounded caching and cancel-isolated batches."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector
from powercontext.builtin.inference.usage import InferenceUsage

logger = logging.getLogger(__name__)


def _sum_usage(usages: list[InferenceUsage]) -> InferenceUsage:
    if not usages:
        return InferenceUsage(requests=0)
    tokens: dict[str, int | None] = {}
    for field in ("input_tokens", "output_tokens"):
        values = [getattr(usage, field) for usage in usages]
        tokens[field] = None if any(value is None for value in values) else sum(int(value) for value in values)
    return InferenceUsage(
        requests=sum(usage.requests for usage in usages),
        input_tokens=tokens["input_tokens"],
        output_tokens=tokens["output_tokens"],
    )


@dataclass(frozen=True)
class _Resolution:
    """One chunk outcome shared by every waiter of its texts."""

    vector: EmbeddingVector
    batch_id: int
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Merge concurrent embedding requests, cache successful vectors, and keep usage per caller."""

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must be non-negative")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[_Resolution]] = {}
        self._inflight: dict[str, asyncio.Future[_Resolution]] = {}
        self._window_handle: asyncio.TimerHandle | None = None
        self._batch_counter = 0
        self._chunk_tasks: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        texts = tuple(texts)
        results: list[EmbeddingVector | None] = [None] * len(texts)
        waited: dict[str, asyncio.Future[_Resolution]] = {}
        async with self._lock:
            for index, text in enumerate(texts):
                if results[index] is not None:
                    continue
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    results[index] = cached
                    continue
                if text in waited:
                    continue
                future = self._pending.get(text) or self._inflight.get(text)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    self._pending[text] = future
                waited[text] = future
            self._ensure_window()
        if waited:
            for future in waited.values():
                await asyncio.shield(future)
            positions: dict[str, list[int]] = {}
            for index, text in enumerate(texts):
                if text in waited:
                    positions.setdefault(text, []).append(index)
            resolutions: dict[int, InferenceUsage] = {}
            for text, future in waited.items():
                resolution = future.result()
                resolutions.setdefault(resolution.batch_id, resolution.usage)
                for index in positions[text]:
                    results[index] = resolution.vector
            usage = _sum_usage(list(resolutions.values()))
        else:
            usage = InferenceUsage(requests=0)
        return EmbeddingResult(vectors=tuple(result for result in results), usage=usage)

    def _ensure_window(self) -> None:
        if not self._pending or self._window_handle is not None:
            return
        if self._coalesce_seconds > 0:
            self._window_handle = asyncio.get_running_loop().call_later(
                self._coalesce_seconds, self._schedule_dispatch
            )
        else:
            self._schedule_dispatch()

    def _schedule_dispatch(self) -> None:
        self._window_handle = None
        task = asyncio.create_task(self._dispatch_window())
        self._chunk_tasks.add(task)
        task.add_done_callback(self._chunk_tasks.discard)

    async def _dispatch_window(self) -> None:
        async with self._lock:
            if not self._pending:
                return
            texts = tuple(self._pending)
            futures = dict(self._pending)
            self._pending.clear()
            self._window_handle = None
            for text, future in futures.items():
                self._inflight[text] = future
        for start in range(0, len(texts), self._max_batch_size):
            chunk = texts[start : start + self._max_batch_size]
            chunk_futures = [futures[text] for text in chunk]
            task = asyncio.create_task(self._run_chunk(tuple(chunk), chunk_futures))
            self._chunk_tasks.add(task)
            task.add_done_callback(self._chunk_tasks.discard)

    async def _run_chunk(self, texts: tuple[str, ...], futures: list[asyncio.Future[_Resolution]]) -> None:
        batch_id = self._batch_counter
        self._batch_counter += 1
        try:
            result = await self._model.embed(texts)
            vectors = tuple(result.vectors)
            if len(vectors) != len(texts):
                raise ValueError(
                    f"embedding model returned {len(vectors)} vectors for {len(texts)} texts"
                )
        except Exception as error:
            async with self._lock:
                for text, future in zip(texts, futures):
                    self._inflight.pop(text, None)
                    if not future.done():
                        future.set_exception(error)
            return
        async with self._lock:
            for text, future, vector in zip(texts, futures, vectors):
                if self._cache_size > 0:
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                self._inflight.pop(text, None)
                if not future.done():
                    future.set_result(_Resolution(vector=vector, batch_id=batch_id, usage=result.usage))
