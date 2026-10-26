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

"""Financial-report PDF parsing with page-level Sources and question retrieval."""

from __future__ import annotations

import hashlib
import io
from urllib.parse import quote, unquote

from pypdf import PdfReader
from pydantic import BaseModel

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

_PDF_PAGE_SOURCE_TYPE = "pdf-page"
_IDENTITY_PREFIX = "pdf-page:"


class PDFDocumentInput(BaseModel):
    """One complete offline PDF report addressed by its document id."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One parsed physical page with its document identity and extracted text."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one PDF report atomically, pages from 1."""

    document_id = value.document_id
    if not document_id or document_id.strip() != document_id:
        raise ValueError("document_id must be non-empty without surrounding whitespace")
    try:
        reader = PdfReader(io.BytesIO(value.content))
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise ValueError("the PDF requires a non-empty password")
        extracted: list[str] = []
        for pdf_page in reader.pages:
            extracted.append(pdf_page.extract_text() or "")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"unreadable PDF document: {error}") from error
    digest = hashlib.sha256(value.content).hexdigest()
    return tuple(
        PDFPageInput(document_id=document_id, document_sha256=digest, page_number=page_number, text=text)
        for page_number, text in enumerate(extracted, start=1)
    )


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable page identity from the document id, digest, and page."""

    identity = f"{_IDENTITY_PREFIX}{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        identity = f"{_IDENTITY_PREFIX}{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
    return identity


def _parse_page_identity(source_id: str) -> tuple[str, int] | None:
    """Recover ``(document_id, page_number)`` from a parseable page Source id."""

    parts = source_id.split(":")
    if len(parts) != 4 or parts[0] != "pdf-page" or not parts[3].isdigit():
        return None
    return unquote(parts[1]), int(parts[3])


class PDFPageSource(Source):
    """One CAPTURED Source holding one parsed physical PDF page."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class _PDFPageTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=_PDF_PAGE_SOURCE_TYPE,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class _PDFPageSourceDefinition:
    name = "pdf-page"
    version = "1"
    input_class = PDFPageInput
    source_class = PDFPageSource
    projections = (_PDFPageTextEvidenceProjection(),)

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            definition_version=self.version,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


PDF_PAGE_SOURCE_DEFINITION = _PDFPageSourceDefinition()


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Search one scoped Memory for distinct evidence pages answering ``question``."""

    if limit <= 0:
        return ()
    query = question.strip()
    if not query:
        return ()
    pool = max(limit * 4, 64)
    page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=pool))
    citations: list[MemoryCitation] = []
    seen: set[tuple[str, int]] = set()
    for hit in page.hits:
        if len(citations) >= limit:
            break
        record = await memory.get(
            GetMemoryEntryRequest(
                citation=MemoryCitation(
                    memory_ref=hit.memory_ref,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
            )
        )
        locations: set[tuple[str, int]] = set()
        for reference in record.entry.sources:
            parsed = _parse_page_identity(reference.source_id)
            if parsed is not None:
                locations.add(parsed)
        if len(locations) != 1:
            continue
        location = next(iter(locations))
        if location in seen:
            continue
        seen.add(location)
        citations.append(
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        )
    return tuple(citations)
