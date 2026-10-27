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
import itertools
import math
import re
from urllib.parse import quote, unquote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime.models import GetMemoryEntryRequest, MemoryEntryRecord, SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

_PDF_PAGE_SOURCE_TYPE = "pdf-page"
_IDENTITY_PREFIX = "pdf-page:"

_SEARCH_POOL_FLOOR = 400
_SEARCH_POOL_MULTIPLIER = 20
_QUERY_TOKEN_PATTERN = re.compile(r"[a-z]+|\d+")
_SEARCH_DIGIT_WEIGHT = 0.3
_SEARCH_HEADING_BONUS = 0.15
_SEARCH_WEAK_COVER_THRESHOLD = 0.35
_SEARCH_WEAK_NEIGHBOR_WINDOW = 8
_SEARCH_TABLE_FILLER = 5
_SEARCH_NARRATIVE_FILLER = 2
_SEARCH_NEIGHBOR_WINDOW = 3
_NUMERIC_HINT_TOKENS = frozenset((
    "ratio",
    "margin",
    "percent",
    "percentage",
    "growth",
    "revenue",
    "income",
    "expense",
    "cost",
    "cash",
    "debt",
    "equity",
    "asset",
    "liability",
    "sales",
    "profit",
    "earnings",
    "dividend",
    "spend",
    "spending",
    "ebitda",
    "ebit",
))
_QUERY_STOP_TOKENS = frozenset((
    "a",
    "an",
    "the",
    "and",
    "or",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "from",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "their",
    "there",
    "than",
    "then",
    "so",
    "such",
    "not",
    "no",
    "but",
    "if",
    "into",
    "about",
    "over",
    "under",
    "between",
    "during",
    "each",
    "any",
    "all",
    "have",
    "has",
    "had",
    "having",
    "he",
    "she",
    "they",
    "them",
    "we",
    "us",
    "our",
    "you",
    "your",
    "i",
    "me",
    "my",
    "also",
    "more",
    "most",
    "much",
    "many",
    "other",
    "able",
))


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
    """Search one scoped Memory for distinct evidence pages answering ``question``.

    Candidates come from the full scoped entry list so retrieval does not
    depend on lexical admission; pages are then ranked offline by
    idf-weighted query-term coverage with a numeric-table bias, so a page
    containing every query term once outranks boilerplate that repeats one
    term and quantity questions can still surface statement tables whose
    vocabulary never overlaps the question.
    """

    if limit <= 0:
        return ()
    query = question.strip()
    if not query:
        return ()
    try:
        listed = await memory.list()
    except Exception:
        listed = None
    if listed is not None and listed.memory_ref is not None:
        records = _scoped_page_records(listed)
        if records:
            ranked = _rank_page_records(records, query, limit)
            return tuple(record.citation for record, _location in ranked[:limit])
    return await _lexical_search_pages(memory, query, limit)


def _scoped_page_records(listed) -> list[tuple[MemoryEntryRecord, tuple[str, int]]]:
    records: list[tuple[MemoryEntryRecord, tuple[str, int]]] = []
    for record in listed.entries:
        locations = _entry_locations(record.entry)
        if len(locations) == 1:
            records.append((record, next(iter(locations))))
    return records


async def _lexical_search_pages(memory, query: str, limit: int) -> tuple[MemoryCitation, ...]:
    pool = max(limit * _SEARCH_POOL_MULTIPLIER, _SEARCH_POOL_FLOOR)
    result = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=pool))
    ranked = _rank_page_hits(result.hits, query)
    citations: list[MemoryCitation] = []
    seen: set[tuple[str, int]] = set()
    for hit in ranked:
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
        locations = _entry_locations(record.entry)
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


def _entry_locations(entry) -> set[tuple[str, int]]:
    locations: set[tuple[str, int]] = set()
    for reference in entry.sources:
        parsed = _parse_page_identity(reference.source_id)
        if parsed is not None:
            locations.add(parsed)
    return locations


def _rank_page_records(records, question: str, limit: int) -> list:
    """Order (record, location) pairs by query-term coverage over all pages.

    Idf comes from the scoped corpus itself, so shared boilerplate (the
    company name on every page) counts little while distinctive terms
    dominate. Numeric-flavored questions additionally favour dense data
    tables, and the tail slots are reserved for the target document's
    densest tables because financial-statement gold pages often share no
    vocabulary with the question.
    """

    token_sets: list[frozenset[str]] = []
    densities: list[float] = []
    documents: list[str] = []
    texts: list[str] = []
    for record, (document_id, _page) in records:
        text = record.entry.text
        prefix = f"{document_id}\n"
        if text.startswith(prefix):
            text = text[len(prefix) :]
        tokens = _QUERY_TOKEN_PATTERN.findall(text.lower())
        stripped = [_strip_plural(token) for token in tokens]
        digits = sum(1 for token in stripped if token.isdigit())
        token_sets.append(frozenset(stripped))
        densities.append(digits / len(stripped) if stripped else 0.0)
        documents.append(document_id)
        texts.append(text)
    total = len(records)
    query_tokens = _query_tokens(question)
    if not query_tokens:
        order = sorted(range(total), key=lambda i: (-densities[i], documents[i], records[i][1][1]))
        return [records[i] for i in order]
    query_set = set(query_tokens)
    numeric = _is_numeric_question(question)
    document_frequency: dict[str, int] = {}
    for tokens in token_sets:
        for token in query_set & tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    idf = {token: math.log(1.0 + total / (1.0 + document_frequency.get(token, 0))) for token in query_set}
    denominator = sum(idf[token] for token in query_tokens) or 1.0
    target = _infer_target_document(documents, token_sets, query_tokens, idf, denominator, question)
    if target is not None:
        keep = [index for index in range(total) if documents[index] == target]
        # Second pass: idf within the target document, so terms that are
        # boilerplate across the whole corpus (the company name) still
        # discriminate between the target document's own pages.
        target_frequency: dict[str, int] = {}
        for index in keep:
            for token in query_set & token_sets[index]:
                target_frequency[token] = target_frequency.get(token, 0) + 1
        idf = {token: math.log(1.0 + len(keep) / (1.0 + target_frequency.get(token, 0))) for token in query_set}
    else:
        keep = list(range(total))
    scored = _score_pages(token_sets, densities, idf, query_tokens, keep, numeric, _heading_bonus(question, texts))
    return _select_page_records(records, scored, keep, densities, target, numeric, limit)


def _infer_target_document(documents, token_sets, query_tokens, idf, denominator, question) -> str | None:
    """Decide the question's target document, or ``None`` when ambiguous.

    Affinity matches the question against the document ids: a document-id
    word that also appears in the flattened question is the strongest
    signal (the question names that company), question tokens matching a
    document-id word or the flattened id weigh less, and adjacent-token
    concatenations catch multi-word names. The per-document mean coverage
    breaks ties. Only a strictly best affinity narrows retrieval to that
    document.
    """

    affinity: dict[str, int] = {}
    flat_question = re.sub(r"[^a-z0-9]", "", question.lower())
    for document_id in set(documents):
        lowered = document_id.lower()
        flat_document = re.sub(r"[^a-z0-9]", "", lowered)
        words = set(re.findall(r"[a-z]+|\d+", lowered))
        score = 0
        for word in words:
            if len(word) >= 2 and not word.isdigit() and not re.fullmatch(r"10[kq]", word) and word in flat_question:
                score += 3
        for token in query_tokens:
            if len(token) >= 3 and re.search(r"[a-z]", token):
                if token in words:
                    score += 2
                elif token in flat_document:
                    score += 1
        for first, second in itertools.pairwise(query_tokens):
            if len(first) + len(second) >= 6 and first + second in flat_document:
                score += 3
        # Company names with no standalone words ("3M_2022_10K" flattens to
        # "3m202210k") survive only as short substrings; a short id prefix
        # occurring in the flattened question is still a naming signal.
        if flat_document[:2] and flat_document[:2] not in flat_document[2:] and flat_document[:2] in flat_question:
            score += 3
        affinity[document_id] = score
    best_affinity = max(affinity.values(), default=0)
    affinity_values = sorted(affinity.values(), reverse=True)
    decisive = best_affinity > 0 and (len(affinity_values) < 2 or affinity_values[0] > affinity_values[1])
    if not decisive:
        return None
    weight_sum: dict[str, float] = {}
    page_count: dict[str, int] = {}
    for index, tokens in enumerate(token_sets):
        coverage = sum(idf[token] for token in query_tokens if token in tokens) / denominator
        weight_sum[documents[index]] = weight_sum.get(documents[index], 0.0) + coverage
        page_count[documents[index]] = page_count.get(documents[index], 0) + 1
    return min(
        (doc for doc, value in affinity.items() if value == best_affinity),
        key=lambda doc: (-(weight_sum[doc] / page_count[doc]), doc),
    )


_STATEMENT_HEADING_SYNONYMS = (
    (
        ("balance sheet", "statement of financial position", "financial position"),
        ("balance sheet",),
    ),
    (
        ("income statement", "statement of income", "statements of income"),
        (
            "statements of operations",
            "statement of operations",
            "consolidated statements of income",
            "consolidated statement of income",
        ),
    ),
)


def _heading_bonus(question: str, page_texts: list[str]) -> list[float]:
    """Reward statement pages when the question names the same statement in
    different words — ``statement of financial position`` vs a page headed
    ``Balance Sheets`` — because primary statements carry the figures the
    question asks for even where the phrasing does not overlap."""

    lowered = question.lower()
    wanted: list[str] = []
    for triggers, targets in _STATEMENT_HEADING_SYNONYMS:
        if any(trigger in lowered for trigger in triggers):
            wanted.extend(targets)
    if not wanted:
        return [0.0] * len(page_texts)
    lowered_texts = [text.lower() for text in page_texts]
    return [
        _SEARCH_HEADING_BONUS if any(target in text for target in wanted) else 0.0
        for text in lowered_texts
    ]


def _score_pages(token_sets, densities, idf, query_tokens, keep, numeric, heading_bonuses) -> list[tuple[float, float, int]]:
    denominator = sum(idf[token] for token in query_tokens) or 1.0
    digit_weight = _SEARCH_DIGIT_WEIGHT if numeric else 0.0
    scored: list[tuple[float, float, int]] = []
    for index in keep:
        coverage = sum(idf[token] for token in query_tokens if token in token_sets[index]) / denominator
        score = coverage + digit_weight * densities[index] + heading_bonuses[index]
        scored.append((-score, -densities[index], index))
    scored.sort()
    return scored


def _select_page_records(records, scored, keep, densities, target, numeric, limit) -> list:
    picked: list[tuple[str, int]] = []
    used_pages: set[int] = set()
    filler = (
        _SEARCH_TABLE_FILLER
        if target is not None and numeric
        else _SEARCH_NARRATIVE_FILLER
        if target is not None
        else 0
    )
    head = limit - filler
    for _neg_score, _neg_density, index in scored:
        if len(picked) >= head:
            break
        picked.append(records[index][1])
        used_pages.add(records[index][1][1])
    if target is not None and len(picked) < limit:
        head_pages = [page for document, page in picked if document == target]
        # Weak lexical anchoring means the question vocabulary never met the
        # gold page; widen the neighborhood so nearby narrative sections
        # (legal proceedings, business overview) can still be returned.
        window = _SEARCH_NEIGHBOR_WINDOW
        if scored and head:
            best_coverage = -scored[0][0]
            if best_coverage < _SEARCH_WEAK_COVER_THRESHOLD:
                window = _SEARCH_WEAK_NEIGHBOR_WINDOW
        picked.extend(_neighbor_filler(records, keep, densities, target, used_pages, head_pages, limit - len(picked), window))
    order = {location: rank for rank, location in enumerate(picked)}
    return sorted(records, key=lambda item: order.get(item[1], len(order)))


def _neighbor_filler(records, keep, densities, target, used_pages, head_pages, slots, window=_SEARCH_NEIGHBOR_WINDOW) -> list[tuple[str, int]]:
    """Pages adjacent to the head picks — closer pages first — because
    financial-statement gold pages often sit next to the pages the query
    vocabulary does reach."""

    neighbors: set[tuple[int, int]] = set()
    for base_page in head_pages:
        for delta in range(1, window + 1):
            neighbors.add((delta, base_page - delta))
            neighbors.add((delta, base_page + delta))
    filler: list[tuple[float, tuple[str, int]]] = []
    for index in keep:
        document, page = records[index][1]
        if document != target or page in used_pages:
            continue
        if any(candidate == page for _delta, candidate in neighbors):
            filler.append((-densities[index], records[index][1]))
    filler.sort()
    return [location for _density, location in filler[:slots]]


def _is_numeric_question(question: str) -> bool:
    lowered = question.lower()
    if re.search(r"\d", lowered) or "%" in lowered:
        return True
    tokens = {_strip_plural(token) for token in _QUERY_TOKEN_PATTERN.findall(lowered)}
    if tokens & _NUMERIC_HINT_TOKENS:
        return True
    return "how much" in lowered or "how many" in lowered


def _strip_plural(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _query_tokens(question: str) -> tuple[str, ...]:
    """Content tokens of the question, plural-normalized to match pages."""

    seen: set[str] = set()
    tokens: list[str] = []
    for token in _QUERY_TOKEN_PATTERN.findall(question.lower()):
        if token in _QUERY_STOP_TOKENS or token in seen:
            continue
        seen.add(token)
        tokens.append(_strip_plural(token))
    return tuple(tokens)


def _rank_page_hits(hits, question: str) -> list:
    """Order candidate hits by idf-weighted query-term coverage over page text.

    Idf comes from the candidate pool itself, so shared boilerplate terms (the
    company name on every page) count little while terms that distinguish the
    requested content dominate. The lexical score is normalized by the total
    query weight so pages covering all terms once beat pages repeating one.
    """

    hits = list(hits)
    if not hits:
        return []
    query_tokens = _query_tokens(question)
    if not query_tokens:
        return sorted(hits, key=lambda hit: (-hit.score, hit.entry_id))
    query_set = set(query_tokens)
    page_tokens: list[frozenset[str]] = []
    document_frequency: dict[str, int] = {}
    for hit in hits:
        tokens = frozenset(_QUERY_TOKEN_PATTERN.findall(hit.text.lower()))
        page_tokens.append(tokens)
        for token in query_set & tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    total = len(hits)
    idf = {token: math.log(1.0 + total / (1.0 + document_frequency.get(token, 0))) for token in query_set}
    denominator = sum(idf[token] for token in query_tokens) or 1.0
    scored: list[tuple[float, float, int, object]] = []
    for index, (hit, tokens) in enumerate(zip(hits, page_tokens, strict=True)):
        coverage = 0.0
        for token in query_tokens:
            if token in tokens:
                coverage += idf[token]
        coverage /= denominator
        scored.append((-coverage, -hit.score, index, hit))
    scored.sort(key=lambda item: item[:3])
    return [item[3] for item in scored]
