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

"""Row-level CSV capture: one CAPTURED Source per primary-key row."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

_CSV_ROW_SOURCE_TYPE = "csv-row"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document addressed by its document id."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One validated CSV data row with its identity components."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """One CAPTURED Source holding a single CSV row with full field fidelity."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate a complete CSV document into row inputs atomically."""

    document_id = value.document_id
    if not document_id or document_id.strip() != document_id:
        raise ValueError("document_id must be non-empty without surrounding whitespace")
    primary_key = value.primary_key
    if not primary_key:
        raise ValueError("primary_key must be non-empty")

    text = value.content.decode("utf-8")
    if text.startswith("﻿"):
        text = text[1:]

    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error

    if not records:
        raise ValueError("empty CSV document")
    header = records[0]
    if not header or any(not field.strip() for field in header):
        raise ValueError("header fields must be non-empty and not pure whitespace")
    if len(set(header)) != len(header):
        raise ValueError("header fields must be unique")
    if primary_key not in header:
        raise ValueError("primary_key must match one header column exactly")
    primary_key_index = header.index(primary_key)
    columns = tuple(header)

    rows: list[CSVRowInput] = []
    seen_primary_key_values: set[str] = set()
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(columns):
            raise ValueError("data record field count must match the header width")
        primary_key_value = record[primary_key_index]
        if not primary_key_value.strip():
            raise ValueError("primary key value must not be empty or pure whitespace")
        if primary_key_value in seen_primary_key_values:
            raise ValueError("primary key values must be unique within the document")
        seen_primary_key_values.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the stable row identity from percent-encoded document and key."""

    if not document_id or document_id.strip() != document_id:
        raise ValueError("document_id must be non-empty without surrounding whitespace")
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("source_id must not exceed 256 characters")
    return identity


class _CSVRowTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = {column: value for column, value in zip(source.columns, source.values)}
        return TextEvidence(
            source_type=_CSV_ROW_SOURCE_TYPE,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


class _CSVRowSourceDefinition:
    name = "csv-row"
    version = "1"
    input_class = CSVRowInput
    source_class = CSVRowSource
    projections = (_CSVRowTextEvidenceProjection(),)

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            definition_version=self.version,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


CSV_ROW_SOURCE_DEFINITION = _CSVRowSourceDefinition()
