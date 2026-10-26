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

"""Context packs and prefix deltas for incremental context transmission."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DEFAULT_CHUNK_BYTES = 512
_DIGEST_PREFIX = "sha256:"


class ContextPackError(ValueError):
    """Raised for context pack protocol or chunking violations."""


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _b64_encode(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _b64_decode(values: object) -> tuple[bytes, ...]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ContextPackError("items must be an array of base64 strings")
    try:
        return tuple(base64.b64decode(value, validate=True) for value in values)
    except (binascii.Error, ValueError) as error:
        raise ContextPackError("items must be valid standard base64") from error


def _require_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(_DIGEST_PREFIX)
        or len(value) != len(_DIGEST_PREFIX) + 64
        or any(character not in "0123456789abcdef" for character in value[len(_DIGEST_PREFIX) :])
    ):
        raise ContextPackError(f"{field} must be a sha256 digest string")
    return value


def _pack_digest(items: tuple[bytes, ...]) -> str:
    canonical = _canonical({"schema": _PACK_SCHEMA, "items": _b64_encode(items)})
    return _DIGEST_PREFIX + hashlib.sha256(canonical).hexdigest()


class ContextPack:
    """One complete context pack with deterministic chunks and self-sized wire bytes."""

    __slots__ = ("_items", "_digest", "_byte_count")

    def __init__(self, items: tuple[bytes, ...], digest: str, byte_count: int) -> None:
        self._items = items
        self._digest = digest
        self._byte_count = byte_count

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextPack):
            return NotImplemented
        return self._items == other._items and self._digest == other._digest

    def __repr__(self) -> str:
        return f"ContextPack(items={len(self._items)}, digest={self._digest!r})"

    @property
    def items(self) -> tuple[bytes, ...]:
        return self._items

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def byte_count(self) -> int:
        return self._byte_count

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        if not isinstance(items, tuple) or any(not isinstance(item, bytes) for item in items):
            raise ContextPackError("items must be a tuple of bytes")
        items = tuple(items)
        digest = _pack_digest(items)
        encoded = _b64_encode(items)
        byte_count = 0
        for _ in range(16):
            wire = _canonical({"schema": _PACK_SCHEMA, "items": encoded, "digest": digest, "byte_count": byte_count})
            if len(wire) == byte_count:
                break
            byte_count = len(wire)
        else:
            raise ContextPackError("byte count failed to converge")
        return cls(items, digest, byte_count)

    @classmethod
    def from_content(cls, content: str | None, /, *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Chunk deterministic UTF-8 content into bounded, reassemblable items."""

        if isinstance(content, bytes):
            text = content.decode("utf-8")
        elif content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            raise ContextPackError("content must be a string or None")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        items: list[bytes] = []
        pending: list[str] = []
        pending_bytes = 0
        for character in text:
            encoded = character.encode("utf-8")
            if pending_bytes + len(encoded) > chunk_bytes:
                if not pending:
                    raise ContextPackError("chunk_bytes is too small for one UTF-8 character")
                items.append("".join(pending).encode("utf-8"))
                pending = [character]
                pending_bytes = len(encoded)
                continue
            pending.append(character)
            pending_bytes += len(encoded)
        if pending:
            items.append("".join(pending).encode("utf-8"))
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        return _canonical({
            "schema": _PACK_SCHEMA,
            "items": _b64_encode(self._items),
            "digest": self._digest,
            "byte_count": self._byte_count,
        })

    @classmethod
    def from_bytes(cls, payload: bytes, /) -> ContextPack:
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContextPackError("pack payload must be canonical JSON") from error
        if not isinstance(decoded, dict) or set(decoded) != {"schema", "items", "digest", "byte_count"}:
            raise ContextPackError("pack payload must contain exactly the protocol fields")
        if decoded["schema"] != _PACK_SCHEMA:
            raise ContextPackError("unknown pack schema")
        items = _b64_decode(decoded["items"])
        digest = _require_digest(decoded["digest"], "digest")
        byte_count = decoded["byte_count"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count != len(payload):
            raise ContextPackError("byte_count must equal the received payload length")
        if digest != _pack_digest(items):
            raise ContextPackError("pack digest mismatch")
        return cls(items, digest, byte_count)


@dataclass(frozen=True)
class ContextPackDelta:
    """One prefix-preserving delta between two complete packs."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        prefix = 0
        for base_item, target_item in zip(base.items, target.items):
            if base_item != target_item:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _b64_encode(self.appended),
        })

    @classmethod
    def from_bytes(cls, payload: bytes, /) -> ContextPackDelta:
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContextPackError("delta payload must be canonical JSON") from error
        if not isinstance(decoded, dict) or set(decoded) != {
            "schema",
            "base_digest",
            "target_digest",
            "prefix_items",
            "appended",
        }:
            raise ContextPackError("delta payload must contain exactly the protocol fields")
        if decoded["schema"] != _DELTA_SCHEMA:
            raise ContextPackError("unknown delta schema")
        prefix = decoded["prefix_items"]
        if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        return cls(
            base_digest=_require_digest(decoded["base_digest"], "base_digest"),
            target_digest=_require_digest(decoded["target_digest"], "target_digest"),
            prefix_items=prefix,
            appended=_b64_decode(decoded["appended"]),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Validate one delta against its base pack and rebuild the target pack."""

    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the provided base pack")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base pack chunk count")
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack digest does not match the delta target digest")
    return target
