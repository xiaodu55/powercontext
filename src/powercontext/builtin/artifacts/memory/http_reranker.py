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

"""HTTP endpoint-backed Memory reranker for the versioned selection protocol."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision, MemoryReranker
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Route coarse Memory candidates through one HTTP scoring endpoint in batches."""

    policy_id = "http-reranker-v1"

    def __init__(self, endpoint: str, /, *, client: httpx.AsyncClient | None = None, batch_size: int = 8) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._endpoint = endpoint
        self._client = client if client is not None else httpx.AsyncClient()
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if not candidates:
            raise ValueError("memory reranking requires at least one candidate")
        if limit < 1 or limit > len(candidates):
            raise ValueError("memory reranking limit must be within the candidate count")

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            payload = {
                "query": query,
                "candidates": [
                    {"rank": start + index + 1, "text": hit.text} for index, hit in enumerate(batch)
                ],
            }
            requests += 1
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            self._collect_scores(response, candidates, scores)

        if len(scores) != len(candidates):
            raise ValueError("reranker scores must cover every candidate rank exactly once")
        ranked = sorted(
            ((rank, hit) for rank, hit in enumerate(candidates, start=1)),
            key=lambda pair: -scores[pair[0]],
        )
        selected_ranks = tuple(rank for rank, _hit in ranked[:limit])
        return MemoryRerankDecision(selected_ranks=selected_ranks, usage=InferenceUsage(requests=requests))

    def _collect_scores(
        self,
        response: httpx.Response,
        candidates: tuple[MemoryHit, ...],
        scores: dict[int, float],
    ) -> None:
        payload = response.json()
        if not isinstance(payload, dict) or set(payload) != {"scores"}:
            raise ValueError("reranker response must contain exactly one 'scores' field")
        items = payload["scores"]
        if not isinstance(items, list):
            raise ValueError("reranker scores must be a list")
        for item in items:
            if not isinstance(item, dict) or set(item) != {"rank", "score"}:
                raise ValueError("each reranker score must contain exactly 'rank' and 'score'")
            rank = item["rank"]
            score = item["score"]
            if type(rank) is not int or not 1 <= rank <= len(candidates):
                raise ValueError("reranker rank must be an integer within the candidate count")
            if rank in scores:
                raise ValueError("reranker ranks must not repeat across batches")
            if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score):
                raise ValueError("reranker scores must be finite numbers")
            scores[rank] = float(score)

    def _dump(self, value: Any) -> Any:
        return json.loads(json.dumps(value))
