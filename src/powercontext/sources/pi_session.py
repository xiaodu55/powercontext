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

"""Pi session branch capture: visible-event Sources with checkpointed recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

_PI_EVENT_SOURCE_TYPE = "pi-event"
_MAX_IDENTITY_LENGTH = MAX_SOURCE_ID_LENGTH


class PiSessionInput(BaseModel):
    """One complete offline Pi session archive with an optional selected leaf."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch with its full input record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, object]


def _visible_blocks(content: object) -> list[str]:
    """Render message content as visible text parts, skipping thinking and images."""

    if content is None:
        return []
    if isinstance(content, str):
        return [content] if content else []
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str) and block["text"]:
            parts.append(block["text"])
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            parts.append(f"[tool_call {name} {call_id}] {rendered}")
    return parts


def _render_event(record: dict[str, object]) -> tuple[str, str] | None:
    """Return ``(role, body)`` for one visible event, or ``None`` when invisible."""

    event_type = record.get("type")
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        if role == "toolResult":
            tool_name = message.get("toolName")
            tool_call_id = message.get("toolCallId")
            body_text = "\n".join(_visible_blocks(message.get("content")))
            return role, f"[tool_result {tool_name} {tool_call_id}] {body_text}"
        if role == "bashExecution":
            command = message.get("command")
            output = message.get("output")
            body_text = "" if output is None else str(output)
            return role, f"$ {command}\n{body_text}"
        body = "\n".join(_visible_blocks(message.get("content")))
        return (role, body) if body else None
    if event_type == "custom_message":
        body = "\n".join(_visible_blocks(record.get("content")))
        return ("custom", body) if body else None
    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if isinstance(summary, str) and summary:
            return str(event_type), summary
        return None
    return None


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse the selected branch's visible events atomically, ordered root to leaf."""

    try:
        text = value.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("session content must be UTF-8") from error
    if text.startswith("﻿"):
        text = text[1:]
    records: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed session JSONL: {error}") from error
        if not isinstance(record, dict):
            raise ValueError("session records must be JSON objects")
        records.append(record)
    if not records:
        raise ValueError("empty session archive")
    header = records[0]
    if header.get("type") != "session":
        raise ValueError("first session record must have type=session")
    if header.get("version") != 3:
        raise ValueError("session version must be 3")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session id must be a non-empty string")

    nodes: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for record in records[1:]:
        node_id = record.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("session records need non-empty string ids")
        if node_id in nodes:
            raise ValueError(f"session record {node_id!r} has a duplicate id")
        parent = record.get("parentId")
        if parent is not None and parent != "":
            if not isinstance(parent, str):
                raise ValueError("parentId must be a string when present")
            if parent not in nodes:
                raise ValueError(f"session record {node_id!r} references unknown parent {parent!r}")
        nodes[node_id] = record
        order.append(node_id)
    if not order:
        raise ValueError("session archive has no events to select")
    leaf = value.leaf_id if value.leaf_id is not None else order[-1]
    if leaf not in nodes:
        raise ValueError(f"selected leaf {leaf!r} is not part of the session")

    path_ids: list[str] = []
    cursor: str | None = leaf
    while cursor is not None:
        path_ids.append(cursor)
        parent = nodes[cursor].get("parentId")
        cursor = parent if isinstance(parent, str) and parent else None
    path_ids.reverse()

    events: list[PiEventInput] = []
    for node_id in path_ids:
        record = nodes[node_id]
        rendered = _render_event(record)
        if rendered is None:
            continue
        role, body = rendered
        raw_parent = record.get("parentId")
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=node_id,
                parent_id=raw_parent if isinstance(raw_parent, str) else None,
                event_type=str(record.get("type") or ""),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable event identity from percent-encoded session and event ids."""

    identity = f"pi-event:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > _MAX_IDENTITY_LENGTH:
        identity = f"pi-event:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
    return identity


def _event_digest(event: PiEventInput) -> str:
    payload = json.dumps(event.record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PiEventSource(Source):
    """One CAPTURED Source holding one raw visible Pi session event."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, object]


class _PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=_PI_EVENT_SOURCE_TYPE,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class _PiEventSourceDefinition:
    name = "pi-event"
    version = "1"
    input_class = PiEventInput
    source_class = PiEventSource
    projections = (_PiEventTextEvidenceProjection(),)

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            definition_version=self.version,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


PI_EVENT_SOURCE_DEFINITION = _PiEventSourceDefinition()


class PiSessionConnector:
    """Incrementally import the selected branch's visible events from one file."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self._leaf_id))
        checkpoint = session.checkpoint if isinstance(session.checkpoint, dict) else {}
        known_raw = checkpoint.get("events")
        known: dict[str, str] = dict(known_raw) if isinstance(known_raw, dict) else {}
        for event in events:
            source_id = pi_event_source_id(event.session_id, event.event_id)
            digest = _event_digest(event)
            previous = known.get(source_id)
            if previous is not None and previous != digest:
                raise ValueError(
                    f"imported pi event {event.event_id!r} was rewritten; repair the session input to continue"
                )
        for event in events:
            source_id = pi_event_source_id(event.session_id, event.event_id)
            if source_id in known:
                continue
            await session.submit(source_id, "pi-event", event)
            known[source_id] = _event_digest(event)
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": 1, "events": known},
        )
