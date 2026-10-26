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

"""Source-withdrawal lifecycle: availability propagation over evidence DAGs."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from powercontext.artifacts import ArtifactRef
from powercontext.sources.models import SourceRef


class SourceLifecycleError(ValueError):
    """Raised for lifecycle registration, event, and resolution violations."""


class SourceLifecycleEvent(BaseModel):
    """One availability transition applied to a registered Source node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    reason: str
    revision: int
    sequence: int
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def validate_non_blank(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def validate_positive(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value


class _LifecycleNode:
    __slots__ = (
        "available",
        "kind",
        "node_id",
        "parents",
        "policy",
        "reference",
        "revision",
        "synced",
    )

    def __init__(
        self,
        node_id: str,
        kind: str,
        reference: dict[str, Any],
        available: bool,
        *,
        policy: str | None = None,
        parents: tuple[str, ...] = (),
        revision: int | None = None,
        synced: bool | None = None,
    ) -> None:
        self.node_id = node_id
        self.kind = kind
        self.policy = policy
        self.parents = parents
        self.reference = reference
        self.revision = revision
        self.available = available
        # Memory nodes only: whether the native entry state currently reflects
        # ``available``. ``None`` means "not confirmed yet" and forces one sync
        # pass so interrupted propagations complete on retry.
        self.synced = synced


def _persist_path(database: Any) -> str | None:
    url = getattr(database, "url", None)
    if not isinstance(url, str) or ":memory:" in url:
        return None
    prefix = "sqlite+aiosqlite:///"
    if url.startswith(prefix):
        return url[len(prefix) :].rstrip("/") + ".lifecycle.json"
    return None


class SourceLifecycleFactory:
    """Own lifecycle state storage and hand out one scoped service per scope."""

    def __init__(self, *, runtime: Any, database: Any) -> None:
        self._runtime = runtime
        self._persist_path = _persist_path(database)
        self._loaded = False
        self._nodes: dict[tuple[str, str], _LifecycleNode] = {}
        self._events: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._services: dict[str, _SourceLifecycleService] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded or self._persist_path is None:
            self._loaded = True
            return
        self._loaded = True
        try:
            with open(self._persist_path, encoding="utf-8") as handle:
                snapshot = json.load(handle)
        except FileNotFoundError:
            return
        for scope_id, payload in snapshot.get("scopes", {}).items():
            for node_id, node_data in payload.get("nodes", {}).items():
                self._nodes[(scope_id, node_id)] = _LifecycleNode(
                    node_id=node_id,
                    kind=node_data["kind"],
                    reference=node_data["reference"],
                    available=node_data["available"],
                    policy=node_data.get("policy"),
                    parents=tuple(node_data.get("parents", ())),
                    revision=node_data.get("revision"),
                    synced=node_data.get("synced"),
                )
            for event_key, event_data in payload.get("events", {}).items():
                node_id, sequence = event_key.rsplit(":", 1)
                self._events[(scope_id, node_id, int(sequence))] = event_data

    def _flush(self) -> None:
        if self._persist_path is None:
            return
        scopes: dict[str, dict[str, Any]] = {}
        for (scope_id, node_id), node in self._nodes.items():
            scopes.setdefault(scope_id, {"nodes": {}, "events": {}})["nodes"][node_id] = {
                "kind": node.kind,
                "policy": node.policy,
                "parents": list(node.parents),
                "reference": node.reference,
                "revision": node.revision,
                "available": node.available,
                "synced": node.synced,
            }
        for (scope_id, node_id, sequence), event in self._events.items():
            scopes.setdefault(scope_id, {"nodes": {}, "events": {}})["events"][f"{node_id}:{sequence}"] = event
        with open(self._persist_path, "w", encoding="utf-8") as handle:
            json.dump({"scopes": scopes}, handle, ensure_ascii=False)

    def for_scope(self, scope_id: str, /) -> _SourceLifecycleService:
        self._ensure_loaded()
        service = self._services.get(scope_id)
        if service is None:
            service = _SourceLifecycleService(self, scope_id)
            self._services[scope_id] = service
        return service

    def topic_artifact_available(self, scope_id: str, artifact_id: str, /) -> bool:
        self._ensure_loaded()
        for (scope, _node_id), node in self._nodes.items():
            if scope == scope_id and node.kind == "topic" and node.reference.get("artifact_id") == artifact_id:
                return node.available
        return True

    def register_node_state(self, scope_id: str, node: _LifecycleNode) -> None:
        self._nodes[(scope_id, node.node_id)] = node

    def node_states(self, scope_id: str) -> dict[str, _LifecycleNode]:
        return {node_id: node for (scope, node_id), node in self._nodes.items() if scope == scope_id}


class _SourceLifecycleService:
    """One scope's dependency DAG with availability propagation."""

    def __init__(self, factory: SourceLifecycleFactory, scope_id: str) -> None:
        self._factory = factory
        self._runtime = factory._runtime
        self.scope_id = scope_id

    # ------------------------------------------------------------------ state

    def _nodes(self) -> dict[str, _LifecycleNode]:
        return self._factory.node_states(self.scope_id)

    def _reference_model(self, node: _LifecycleNode) -> Any:
        reference = dict(node.reference)
        if node.kind == "source":
            return SourceRef.model_validate(reference)
        if node.kind == "memory":
            from powercontext.builtin.artifacts.memory.models import MemoryCitation

            return MemoryCitation.model_validate(reference)
        return ArtifactRef.model_validate(reference)

    # -------------------------------------------------------------- validation

    async def _verify_source(self, source_ref: SourceRef, /) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(
                source_ref.source_type, source_ref.source_id
            )
        except Exception as error:
            raise SourceLifecycleError("source reference is not readable") from error

    async def _verify_memory_citation(self, citation: Any, /) -> None:
        from powercontext.builtin.runtime.application import (
            ModelUsagePurpose,
            _head_or_none,
        )

        async with (
            self._runtime._context(self.scope_id, embedding_purpose=ModelUsagePurpose.MEMORY_INDEXING) as context,
            self._runtime._locked(self.scope_id),
        ):
            service = context.artifacts.memory
            current = await _head_or_none(service, context.artifacts.memory_artifact_id)
            if current is None:
                raise SourceLifecycleError("memory artifact is not readable")
            if all(item.entry_id != citation.entry_id for item in current.content.manifest.entries):
                raise SourceLifecycleError(f"memory entry {citation.entry_id!r} is not readable")

    async def _verify_topic_reference(self, artifact_ref: ArtifactRef, /) -> None:
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                artifact_ref.family, artifact_ref.artifact_id, artifact_ref.revision
            )
        except Exception as error:
            raise SourceLifecycleError(f"topic artifact {artifact_ref.artifact_id!r} is not readable") from error

    # ----------------------------------------------------------- registration

    async def register_source(self, node_id: str, source_ref: SourceRef, /, *, revision: int) -> None:
        self._factory._ensure_loaded()
        if revision <= 0:
            raise SourceLifecycleError("source lifecycle revision must be positive")
        await self._verify_source(source_ref)
        nodes = self._nodes()
        existing = nodes.get(node_id)
        reference = source_ref.model_dump(mode="json")
        if existing is not None:
            if existing.kind != "source" or existing.reference != reference or existing.revision != revision:
                raise SourceLifecycleError(f"node {node_id!r} is already registered with a different binding")
            return
        node = _LifecycleNode(
            node_id=node_id,
            kind="source",
            reference=reference,
            available=True,
            revision=revision,
        )
        self._factory.register_node_state(self.scope_id, node)
        self._factory._flush()

    async def register_memory(
        self,
        node_id: str,
        citation: Any,
        /,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        await self._verify_memory_citation(citation)
        await self._register_derived(node_id, "memory", citation.model_dump(mode="json"), parents, policy)
        node = self._nodes()[node_id]
        await self._sync_memory_node(node, reason="registered without full support")
        self._factory._flush()

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        await self._verify_topic_reference(artifact_ref)
        await self._register_derived(node_id, "topic", artifact_ref.model_dump(mode="json"), parents, policy)
        self._factory._flush()

    async def _register_derived(
        self,
        node_id: str,
        kind: str,
        reference: dict[str, Any],
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        self._factory._ensure_loaded()
        nodes = self._nodes()
        existing = nodes.get(node_id)
        if existing is not None:
            if (
                existing.kind != kind
                or existing.reference != reference
                or existing.parents != tuple(parents)
                or existing.policy != policy
            ):
                raise SourceLifecycleError(f"node {node_id!r} is already registered with a different binding")
            return
        if not parents:
            raise SourceLifecycleError("derived nodes require at least one parent")
        if node_id in parents:
            raise SourceLifecycleError("a node cannot be its own parent")
        if len(set(parents)) != len(parents):
            raise SourceLifecycleError("parents must not repeat")
        for parent in parents:
            if parent not in nodes:
                raise SourceLifecycleError(f"parent {parent!r} is not registered")
        for parent in parents:
            ancestor = parent
            visited = {node_id}
            while True:
                if ancestor in visited:
                    raise SourceLifecycleError("dependency graph must be acyclic")
                visited.add(ancestor)
                node = nodes[ancestor]
                if not node.parents:
                    break
                if node_id in node.parents:
                    raise SourceLifecycleError("dependency graph must be acyclic")
                ancestor = node.parents[0]
        if policy not in {"all", "any"}:
            raise SourceLifecycleError("policy must be 'all' or 'any'")

        parent_nodes = [nodes[parent] for parent in parents]
        support = [node.available for node in parent_nodes]
        available = all(support) if policy == "all" else any(support)
        node = _LifecycleNode(
            node_id=node_id,
            kind=kind,
            reference=reference,
            available=available,
            policy=policy,
            parents=tuple(parents),
        )
        self._factory.register_node_state(self.scope_id, node)

    # ------------------------------------------------------------ propagation

    def _compute_flips(self, nodes: dict[str, _LifecycleNode]) -> list[_LifecycleNode]:
        """Recompute derived availability topologically; return nodes to flip."""

        remaining = {node_id: node for node_id, node in nodes.items() if node.kind != "source"}
        resolved: dict[str, bool] = {
            node_id: node.available for node_id, node in nodes.items() if node.kind == "source"
        }
        flips: list[tuple[_LifecycleNode, bool]] = []
        while remaining:
            progress = False
            for node_id in sorted(remaining):
                node = remaining[node_id]
                if all(parent in resolved for parent in node.parents):
                    support = [resolved[parent] for parent in node.parents]
                    computed = all(support) if node.policy == "all" else any(support)
                    resolved[node_id] = computed
                    if node.available != computed:
                        flips.append((node, computed))
                    del remaining[node_id]
                    progress = True
            if not progress:
                raise SourceLifecycleError("dependency graph must be acyclic")
        return flips

    async def _sync_memory_node(self, node: _LifecycleNode, /, *, reason: str) -> None:
        """Bring one memory node's native entry state in line with its availability.

        Skips nodes that are already confirmed in sync so replaying a successful
        event never creates new native Memory revisions; re-issues the native
        write for nodes an interrupted run left behind.
        """

        if node.kind != "memory" or node.synced is node.available:
            return
        citation = self._reference_model(node)
        await self._apply_memory_state(citation, active=node.available, reason=reason)
        node.synced = node.available

    async def _apply_memory_state(self, citation: Any, /, *, active: bool, reason: str) -> None:
        from powercontext.builtin.runtime.application import (
            ModelUsagePurpose,
            _head_or_none,
        )

        async with self._runtime._context(
            self.scope_id, embedding_purpose=ModelUsagePurpose.MEMORY_INDEXING
        ) as context, self._runtime._locked(self.scope_id):
            service = context.artifacts.memory
            current = await _head_or_none(service, context.artifacts.memory_artifact_id)
            if current is None:
                return
            target_state = "active" if active else "inactive"
            manifest_entry = next(
                (item for item in current.content.manifest.entries if item.entry_id == citation.entry_id),
                None,
            )
            if manifest_entry is None or manifest_entry.state == target_state:
                return
            entries = await service.entries(current)
            entry = next(item for item in entries if item.entry_id == citation.entry_id)
            if active:
                await service.reactivate(current, entries=(entry,), reason=reason)
            else:
                await service.forget(current, entries=(entry,), reason=reason)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        self._factory._ensure_loaded()
        nodes = self._nodes()
        node = nodes.get(event.node_id)
        if node is None or node.kind != "source":
            raise SourceLifecycleError(f"source {event.node_id!r} is not registered")
        if node.revision != event.revision:
            raise SourceLifecycleError(
                f"source {event.node_id!r} lifecycle revision {event.revision} does not match {node.revision}"
            )
        event_key = (self.scope_id, event.node_id, event.sequence)
        applied = self._factory._events.get(event_key)
        if applied is not None:
            if (
                applied["revision"] != event.revision
                or applied["available"] != event.available
                or applied["reason"] != event.reason
            ):
                raise SourceLifecycleError("event sequence was already applied with different content")
            # Idempotent replay: logical state stays as recorded. Native writes that
            # a previous interrupted run left behind are completed by the sync loop
            # below, while fully synced nodes produce no native calls (and therefore
            # no new Memory revisions).
        else:
            for (scope, node_id, sequence), record in self._factory._events.items():
                if scope == self.scope_id and node_id == event.node_id and sequence >= event.sequence:
                    raise SourceLifecycleError("event sequence must increase monotonically")
            self._factory._events[event_key] = {
                "revision": event.revision,
                "available": event.available,
                "reason": event.reason,
            }
            node.available = event.available
            flips = self._compute_flips(nodes)
            for flipped, computed in flips:
                nodes[flipped.node_id].available = computed
            self._factory._flush()
        for node_id in sorted(nodes):
            await self._sync_memory_node(nodes[node_id], reason=event.reason)
        self._factory._flush()
        return sorted(node_id for node_id, node in nodes.items() if node.available)

    # --------------------------------------------------------------- resolve

    async def resolve(self, node_id: str, /, *, purpose: str) -> Any:
        self._factory._ensure_loaded()
        node = self._nodes().get(node_id)
        if node is None:
            raise SourceLifecycleError(f"node {node_id!r} is not registered")
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError(f"unknown lifecycle resolution purpose {purpose!r}")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError(f"node {node_id!r} is not available for generation")
        return self._reference_model(node)


class LifecycleTopicMemoryProxy:
    """Filter native Topic Memory search results by lifecycle availability."""

    def __init__(self, inner: Any, factory: SourceLifecycleFactory) -> None:
        self._inner = inner
        self._factory = factory

    def for_scope(self, scope_id: str, /) -> _LifecycleScopedTopicMemory:
        return _LifecycleScopedTopicMemory(self._inner.for_scope(scope_id), self._factory, scope_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _LifecycleScopedTopicMemory:
    def __init__(self, inner: Any, factory: SourceLifecycleFactory, scope_id: str) -> None:
        self._inner = inner
        self._factory = factory
        self._scope_id = scope_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def search(self, request: Any, /, **kwargs: Any) -> Any:
        self._factory._ensure_loaded()
        page = await self._inner.search(request, **kwargs)
        hidden = {
            node.reference.get("artifact_id")
            for node in self._factory.node_states(self._scope_id).values()
            if node.kind == "topic" and not node.available
        }
        if not hidden:
            return page
        hits = [hit for hit in page.hits if hit.artifact_ref.artifact_id not in hidden]
        return page.model_copy(update={"hits": hits})
