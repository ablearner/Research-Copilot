import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Any

from adapters.embedding.base import BaseEmbeddingAdapter
from adapters.graph_store.base import BaseGraphStore
from adapters.llm.base import BaseLLMAdapter
from adapters.vector_store.base import BaseVectorStore
from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.document import DocumentPage, ParsedDocument, TextBlock
from domain.schemas.embedding import EmbeddingVector, MultimodalEmbeddingRecord
from domain.schemas.evidence import Evidence, EvidenceBundle
from domain.schemas.graph import GraphEdge, GraphExtractionResult, GraphNode, GraphQueryRequest, GraphQueryResult, GraphTriple
from domain.schemas.retrieval import RetrievalHit
from retrieval.lexical import bm25_score_texts
from rag_runtime.services.layout_service import LayoutService
from rag_runtime.services.ocr_service import OcrService
from rag_runtime.services.pdf_service import PdfService


_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
logger = logging.getLogger(__name__)


class LocalHashEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(self, model: str = "local-hash-embedding", dimensions: int = 128) -> None:
        super().__init__(max_retries=0)
        self.model = model
        self.dimensions = dimensions

    async def _embed_text(self, text: str) -> EmbeddingVector:
        return self._vectorize(text)

    async def _embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        return [self._vectorize(text) for text in texts]

    async def _embed_image(self, image_path: str) -> EmbeddingVector:
        return self._vectorize(Path(image_path).name)

    async def _embed_page(self, page_image_path: str, page_text: str) -> EmbeddingVector:
        return self._vectorize(f"{Path(page_image_path).name}\n{page_text}")

    async def _embed_chart(self, chart_image_path: str, chart_summary: str) -> EmbeddingVector:
        return self._vectorize(f"{Path(chart_image_path).name}\n{chart_summary}")

    def _vectorize(self, text: str) -> EmbeddingVector:
        values = [0.0] * self.dimensions
        for token in _TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return EmbeddingVector(
            model=self.model,
            dimensions=self.dimensions,
            values=[value / norm for value in values],
        )


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.records: dict[str, MultimodalEmbeddingRecord] = {}

    async def upsert_embedding(self, record: MultimodalEmbeddingRecord) -> None:
        self.records[record.id] = record

    async def upsert_embeddings(self, records: list[MultimodalEmbeddingRecord]) -> None:
        for record in records:
            self.records[record.id] = record

    async def search_by_vector(
        self,
        vector: EmbeddingVector,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        filters = filters or {}
        document_ids = set(filters.get("document_ids") or [])
        modalities = set(filters.get("modalities") or [])
        hits: list[RetrievalHit] = []
        for record in self.records.values():
            if document_ids and record.item.document_id not in document_ids:
                continue
            if modalities and record.modality not in modalities:
                continue
            score = self._cosine(vector.values, record.embedding.values)
            hits.append(
                RetrievalHit(
                    id=record.id,
                    source_type=record.item.source_type,
                    source_id=record.item.source_id,
                    document_id=record.item.document_id,
                    content=record.item.content,
                    vector_score=max(score, 0.0),
                    metadata={**record.item.metadata, **record.metadata, "uri": record.item.uri},
                )
            )
        return sorted(hits, key=lambda item: item.vector_score or 0.0, reverse=True)[:top_k]

    async def search_similar_text(self, text: str, top_k: int) -> list[RetrievalHit]:
        query = LocalHashEmbeddingAdapter()._vectorize(text)
        return await self.search_by_vector(query, top_k)

    async def search_sparse_text(
        self,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        filters = filters or {}
        document_ids = set(filters.get("document_ids") or [])
        source_types = set(filters.get("source_types") or [])
        candidates: list[MultimodalEmbeddingRecord] = []
        for record in self.records.values():
            if document_ids and record.item.document_id not in document_ids:
                continue
            if source_types and record.item.source_type not in source_types:
                continue
            if not (record.item.content or "").strip():
                continue
            candidates.append(record)

        scores = bm25_score_texts(
            query=text,
            texts=[record.item.content or "" for record in candidates],
        )
        hits: list[RetrievalHit] = []
        for record, score in zip(candidates, scores, strict=True):
            if score <= 0:
                continue
            hits.append(
                RetrievalHit(
                    id=record.id,
                    source_type=record.item.source_type,
                    source_id=record.item.source_id,
                    document_id=record.item.document_id,
                    content=record.item.content,
                    sparse_score=score,
                    metadata={**record.item.metadata, **record.metadata, "uri": record.item.uri},
                )
            )
        return sorted(hits, key=lambda item: item.sparse_score or 0.0, reverse=True)[:top_k]

    async def delete_by_doc_id(self, doc_id: str) -> None:
        self.records = {
            record_id: record
            for record_id, record in self.records.items()
            if record.item.document_id != doc_id
        }

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        length = min(len(left), len(right))
        return sum(left[index] * right[index] for index in range(length))


class InMemoryGraphStore(BaseGraphStore):
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_nodes(self, nodes: list[GraphNode]) -> None:
        for node in nodes:
            self.nodes[node.id] = node

    async def upsert_edges(self, edges: list[GraphEdge]) -> None:
        for edge in edges:
            self.edges[edge.id] = edge

    async def upsert_triples(self, triples: list[GraphTriple]) -> None:
        await self.upsert_nodes([triple.subject for triple in triples] + [triple.object for triple in triples])
        await self.upsert_edges([triple.predicate for triple in triples])

    async def query_subgraph(self, query_request: GraphQueryRequest) -> GraphQueryResult:
        terms = [term.lower() for term in _TOKEN_PATTERN.findall(query_request.query)]
        document_ids = set(query_request.document_ids)
        nodes = [
            node
            for node in self.nodes.values()
            if self._matches_node(node, terms, document_ids)
        ][: query_request.limit]
        node_ids = {node.id for node in nodes}
        edges = [
            edge
            for edge in self.edges.values()
            if edge.source_node_id in node_ids or edge.target_node_id in node_ids
        ][: query_request.limit]
        triples = self._triples_for_edges(edges)
        evidences = [node.source_reference for node in nodes] + [edge.source_reference for edge in edges]
        return GraphQueryResult(
            query=query_request.query,
            nodes=nodes,
            edges=edges,
            triples=triples,
            evidences=evidences,
        )

    async def get_neighbors(self, node_id: str, depth: int) -> GraphQueryResult:
        edges = [
            edge
            for edge in self.edges.values()
            if edge.source_node_id == node_id or edge.target_node_id == node_id
        ]
        node_ids = {node_id, *[edge.source_node_id for edge in edges], *[edge.target_node_id for edge in edges]}
        nodes = [node for key, node in self.nodes.items() if key in node_ids]
        return GraphQueryResult(query=f"neighbors:{node_id}", nodes=nodes, edges=edges, triples=self._triples_for_edges(edges))

    async def search_entities(
        self,
        keyword: str,
        document_ids: list[str] | None = None,
    ) -> GraphQueryResult:
        return await self.query_subgraph(
            GraphQueryRequest(
                query=keyword,
                document_ids=list(document_ids or []),
                limit=20,
            )
        )

    def _matches_node(self, node: GraphNode, terms: list[str], document_ids: set[str]) -> bool:
        if document_ids and node.source_reference.document_id not in document_ids:
            return False
        text = " ".join([node.label, *[str(value) for value in node.properties.values()]]).lower()
        return not terms or any(term in text for term in terms)

    def _triples_for_edges(self, edges: list[GraphEdge]) -> list[GraphTriple]:
        triples: list[GraphTriple] = []
        for edge in edges:
            source = self.nodes.get(edge.source_node_id)
            target = self.nodes.get(edge.target_node_id)
            if source and target:
                triples.append(GraphTriple(subject=source, predicate=edge, object=target))
        return triples


class LocalDocumentParser(PdfService, OcrService, LayoutService):
    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

    def __init__(self, storage_root: str | Path | None = None) -> None:
        default_root = Path(__file__).resolve().parents[1] / ".data" / "storage"
        self.storage_root = Path(storage_root).expanduser().resolve() if storage_root else default_root.resolve()

    async def parse_document(self, file_path: str, document_id: str | None = None) -> ParsedDocument:
        path = Path(file_path)
        doc_id = document_id or f"doc_{hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:12]}"
        pages = self._parse_pages(path, doc_id)
        return ParsedDocument(
            id=doc_id,
            filename=path.name,
            content_type=self._content_type(path),
            status="parsed",
            pages=pages,
            metadata={
                "parser": self.__class__.__name__,
                "source_path": str(path),
                "page_count": len(pages),
            },
        )

    async def extract_text_blocks(self, page: DocumentPage) -> list[TextBlock]:
        return page.text_blocks

    async def locate_chart_candidates(self, page: DocumentPage) -> list[dict]:
        return []

    def _parse_pages(self, path: Path, document_id: str) -> list[DocumentPage]:
        suffix = path.suffix.lower()
        if suffix in self._IMAGE_SUFFIXES:
            return [self._build_image_page(path, document_id)]
        if suffix == ".pdf":
            return self._build_pdf_pages(path, document_id)
        return [self._build_text_page(path, document_id)]

    def _build_image_page(self, path: Path, document_id: str) -> DocumentPage:
        text = (
            f"Uploaded image file: {path.name}. Use Chart Understand for visual chart analysis; "
            "document text parsing is not available for standalone images in this local parser."
        )
        return DocumentPage(
            id=f"{document_id}_p1",
            document_id=document_id,
            page_number=1,
            image_uri=str(path),
            text_blocks=self._build_text_blocks(
                document_id=document_id,
                page_id=f"{document_id}_p1",
                page_number=1,
                text=text,
            ),
            metadata={"source_type": "image"},
        )

    def _build_pdf_pages(self, path: Path, document_id: str) -> list[DocumentPage]:
        candidate_pages: list[DocumentPage] = []
        for builder in (self._build_pdf_pages_with_pymupdf, self._build_pdf_pages_with_pypdf):
            try:
                pages = builder(path, document_id)
            except Exception:
                continue
            if not pages:
                continue
            candidate_pages = pages
            if any(page.text_blocks or page.image_uri for page in pages):
                return pages

        if candidate_pages and any(page.text_blocks or page.image_uri for page in candidate_pages):
            return candidate_pages
        return [self._build_fallback_page(path, document_id, reason="pdf_no_extractable_text")]

    def _build_pdf_pages_with_pymupdf(self, path: Path, document_id: str) -> list[DocumentPage]:
        import fitz

        try:
            document = fitz.open(str(path))
        except Exception as exc:
            raise RuntimeError("pymupdf_open_failed") from exc

        pages: list[DocumentPage] = []
        try:
            for page_number, pdf_page in enumerate(document, start=1):
                page_id = f"{document_id}_p{page_number}"
                extracted_text = self._normalize_extracted_text(pdf_page.get_text("text", sort=True) or "")
                blocks = self._build_text_blocks(
                    document_id=document_id,
                    page_id=page_id,
                    page_number=page_number,
                    text=extracted_text,
                )
                rect = pdf_page.rect
                page_image_uri = self._render_pdf_page_image(pdf_page=pdf_page, document_id=document_id, page_id=page_id)
                pages.append(
                    DocumentPage(
                        id=page_id,
                        document_id=document_id,
                        page_number=page_number,
                        width=float(rect.width) if rect else None,
                        height=float(rect.height) if rect else None,
                        image_uri=page_image_uri,
                        text_blocks=blocks,
                        metadata={
                            "source_type": "pdf",
                            "text_extracted": bool(extracted_text.strip()),
                            "text_block_count": len(blocks),
                            "text_engine": "pymupdf",
                            "page_image_generated": bool(page_image_uri),
                        },
                    )
                )
        finally:
            document.close()

        if not pages:
            return [self._build_fallback_page(path, document_id, reason="pymupdf_pdf_has_no_pages")]
        return pages

    def _render_pdf_page_image(self, *, pdf_page: Any, document_id: str, page_id: str) -> str | None:
        try:
            import fitz

            target_path = self.storage_root / "page_images" / document_id / f"{page_id}.png"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pixmap.save(str(target_path))
            return str(target_path)
        except Exception:
            logger.warning(
                "Failed to render PDF page image",
                extra={"document_id": document_id, "page_id": page_id},
                exc_info=True,
            )
            return None

    def _build_pdf_pages_with_pypdf(self, path: Path, document_id: str) -> list[DocumentPage]:
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise RuntimeError("pypdf_open_failed") from exc

        pages: list[DocumentPage] = []
        for page_number, pdf_page in enumerate(reader.pages, start=1):
            page_id = f"{document_id}_p{page_number}"
            extracted_text = self._normalize_extracted_text(pdf_page.extract_text() or "")
            blocks = self._build_text_blocks(
                document_id=document_id,
                page_id=page_id,
                page_number=page_number,
                text=extracted_text,
            )
            width, height = self._page_size(pdf_page)
            pages.append(
                DocumentPage(
                    id=page_id,
                    document_id=document_id,
                    page_number=page_number,
                    width=width,
                    height=height,
                    text_blocks=blocks,
                    metadata={
                        "source_type": "pdf",
                        "text_extracted": bool(extracted_text.strip()),
                        "text_block_count": len(blocks),
                        "text_engine": "pypdf",
                    },
                )
            )

        if not pages:
            return [self._build_fallback_page(path, document_id, reason="pypdf_pdf_has_no_pages")]
        return pages

    def _build_text_page(self, path: Path, document_id: str) -> DocumentPage:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        if not text.strip():
            return self._build_fallback_page(path, document_id, reason="text_file_empty")
        return DocumentPage(
            id=f"{document_id}_p1",
            document_id=document_id,
            page_number=1,
            text_blocks=self._build_text_blocks(
                document_id=document_id,
                page_id=f"{document_id}_p1",
                page_number=1,
                text=text,
            ),
            metadata={"source_type": "text"},
        )

    def _build_fallback_page(self, path: Path, document_id: str, reason: str) -> DocumentPage:
        message = f"Uploaded file: {path.name}. Extractable text was not available for this document."
        return DocumentPage(
            id=f"{document_id}_p1",
            document_id=document_id,
            page_number=1,
            text_blocks=self._build_text_blocks(
                document_id=document_id,
                page_id=f"{document_id}_p1",
                page_number=1,
                text=message,
            ),
            metadata={"source_type": "fallback", "reason": reason},
        )

    def _build_text_blocks(
        self,
        *,
        document_id: str,
        page_id: str,
        page_number: int,
        text: str,
    ) -> list[TextBlock]:
        normalized = self._normalize_extracted_text(text)
        if not normalized:
            return []
        return [
            TextBlock(
                id=f"{page_id}_tb_{index + 1}",
                document_id=document_id,
                page_id=page_id,
                page_number=page_number,
                text=chunk,
            )
            for index, chunk in enumerate(self._chunks(normalized))
        ]

    def _normalize_extracted_text(self, text: str) -> str:
        collapsed = text.replace("\x00", " ")
        collapsed = collapsed.replace("\r\n", "\n").replace("\r", "\n")
        collapsed = re.sub(r"[ \t]+", " ", collapsed)
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        return collapsed.strip()

    def _page_size(self, pdf_page: Any) -> tuple[float | None, float | None]:
        media_box = getattr(pdf_page, "mediabox", None)
        if media_box is None:
            return None, None
        try:
            width = float(media_box.width)
            height = float(media_box.height)
            return width, height
        except Exception:
            return None, None

    def _chunks(self, text: str, size: int = 1500, overlap: int = 200) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks: list[str] = []
        for paragraph in paragraphs or [text.strip()]:
            start = 0
            while start < len(paragraph):
                end = start + size
                chunk = paragraph[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end >= len(paragraph):
                    break
                start = end - overlap
        return chunks or [""]

    def _content_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "application/pdf"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix == ".webp":
            return "image/webp"
        if suffix in {".tif", ".tiff"}:
            return "image/tiff"
        return "text/plain"


class LocalLLMAdapter(BaseLLMAdapter):
    def __init__(self) -> None:
        super().__init__(max_retries=0)

    async def _generate_structured(self, prompt: str, input_data: dict[str, Any], response_model: type):
        if response_model is QAResponse:
            bundle = EvidenceBundle.model_validate(input_data.get("evidence_bundle") or {})
            question = str(input_data.get("question", "") or "")
            return self._build_local_qa_response(question=question, evidence_bundle=bundle)
        return response_model.model_validate(input_data)

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        if response_model is ChartSchema:
            path = Path(image_path)
            return ChartSchema(
                id=path.stem or "chart",
                document_id="",
                page_id="",
                page_number=1,
                chart_type="unknown",
                title=path.stem or None,
                summary=f"Local runtime registered chart image {path.name}; visual LLM analysis is not configured.",
                confidence=0.2,
            )
        return response_model()

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        return response_model()

    async def _extract_graph_triples(self, prompt: str, input_data: dict[str, Any], response_model: type):
        document_id = input_data.get("document_id") or "document"
        blocks = input_data.get("text_blocks") or []
        nodes: list[GraphNode] = []
        for index, block in enumerate(blocks[:20]):
            text = str(block.get("text") or "").strip()
            if not text:
                continue
            evidence = Evidence(
                id=f"ev_{block.get('id') or index}",
                document_id=document_id,
                page_id=block.get("page_id"),
                page_number=block.get("page_number"),
                source_type="text_block",
                source_id=block.get("id"),
                snippet=text[:300],
            )
            nodes.append(
                GraphNode(
                    id=f"node_{block.get('id') or index}",
                    label="TextBlock",
                    properties={"name": text[:80], "document_id": document_id},
                    source_reference=evidence,
                )
            )
        return GraphExtractionResult(document_id=document_id, nodes=nodes, status="succeeded", metadata={"llm_provider": "local"})

    def _build_local_qa_response(self, *, question: str, evidence_bundle: EvidenceBundle) -> QAResponse:
        language = "zh" if self._contains_cjk(question) else "en"
        ranked_sentences = self._rank_evidence_sentences(question=question, evidence_bundle=evidence_bundle)
        if not ranked_sentences:
            return QAResponse(
                answer="证据不足" if language == "zh" else "Insufficient evidence.",
                question=question,
                evidence_bundle=evidence_bundle,
                confidence=0.0,
                metadata={"llm_provider": "local", "strategy": "extractive_fallback"},
            )

        top_sentences = ranked_sentences[:3]
        answer = self._compose_local_answer(question=question, sentences=top_sentences, language=language)
        confidence = 0.5 + 0.06 * len(top_sentences)
        if evidence_bundle.summary:
            confidence += 0.05
        if top_sentences and top_sentences[0]["score"] >= 1.0:
            confidence += 0.05
        return QAResponse(
            answer=answer,
            question=question,
            evidence_bundle=evidence_bundle,
            confidence=min(confidence, 0.72),
            metadata={
                "llm_provider": "local",
                "strategy": "extractive_fallback",
                "candidate_count": len(ranked_sentences),
            },
        )

    def _rank_evidence_sentences(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        if evidence_bundle.summary:
            summary_text = self._normalize_text(evidence_bundle.summary)
            if summary_text:
                candidates.append({"text": summary_text, "base_score": 0.95, "source": "summary"})

        for evidence_index, evidence in enumerate(evidence_bundle.evidences[:8]):
            snippet = self._normalize_text(evidence.snippet or "")
            if not snippet:
                continue
            base_score = 0.68 + min(float(evidence.score or 0.0), 1.0) * 0.18 - evidence_index * 0.03
            for sentence_index, sentence in enumerate(self._split_candidate_text(snippet)[:4]):
                normalized = self._dedupe_key(sentence)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append(
                    {
                        "text": sentence,
                        "base_score": max(base_score - sentence_index * 0.05, 0.2),
                        "source": evidence.source_type,
                    }
                )

        english_keywords = self._english_keywords(question)
        chinese_keywords = self._chinese_keywords(question)
        question_chars = self._meaningful_cjk_chars(question)
        summary_question = self._is_summary_question(question)

        ranked: list[dict[str, Any]] = []
        for candidate in candidates:
            text = candidate["text"]
            lowered = text.lower()
            score = float(candidate["base_score"])
            score += 0.22 * sum(1 for token in english_keywords if token in lowered)
            score += 0.28 * sum(1 for token in chinese_keywords if token in text)
            if question_chars:
                score += 0.03 * len(question_chars & self._meaningful_cjk_chars(text))
            if summary_question and candidate["source"] == "summary":
                score += 0.22
            if 12 <= len(text) <= 140:
                score += 0.05
            ranked.append({"text": text, "score": score})

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _compose_local_answer(
        self,
        *,
        question: str,
        sentences: list[dict[str, Any]],
        language: str,
    ) -> str:
        texts = [self._ensure_sentence_end(item["text"], language=language) for item in sentences if item.get("text")]
        if not texts:
            return "证据不足" if language == "zh" else "Insufficient evidence."

        lead = self._trim_sentence_end(texts[0], language=language)
        is_summary_question = self._is_summary_question(question)
        if language == "zh":
            lines = [
                (
                    f"根据当前检索到的证据，这份文档主要讲：{lead}。"
                    if is_summary_question
                    else f"根据当前证据，{lead}。"
                )
            ]
            for index, text in enumerate(texts[1:], start=1):
                prefix = f"{index}. " if is_summary_question else f"补充信息{index}："
                lines.append(f"{prefix}{text}")
            return "\n".join(lines)

        lines = [
            (
                f"Based on the retrieved evidence, the document mainly discusses {lead}."
                if is_summary_question
                else f"Based on the available evidence, {lead}."
            )
        ]
        for index, text in enumerate(texts[1:], start=1):
            prefix = f"{index}. " if is_summary_question else f"Additional detail {index}: "
            lines.append(f"{prefix}{text}")
        return "\n".join(lines)

    def _split_candidate_text(self, text: str) -> list[str]:
        parts = re.split(r"(?:\r?\n)+|(?<=[。！？；.!?;])\s*", text)
        sentences: list[str] = []
        for part in parts:
            cleaned = self._normalize_text(part)
            if len(cleaned) < 6:
                continue
            sentences.append(cleaned[:220])
        return sentences or ([self._normalize_text(text[:220])] if self._normalize_text(text[:220]) else [])

    def _normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return cleaned.strip(" -\t•")

    def _dedupe_key(self, text: str) -> str:
        return re.sub(r"[\s，。！？；,.!?;:：]+", "", text).lower()

    def _contains_cjk(self, text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _english_keywords(self, question: str) -> list[str]:
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "tell",
            "about",
            "document",
            "this",
            "that",
        }
        tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9_]+", question)]
        return [token for token in tokens if len(token) > 2 and token not in stopwords]

    def _chinese_keywords(self, question: str) -> list[str]:
        cleaned = question
        for marker in (
            "这份文档",
            "这个文档",
            "文档里",
            "文档中",
            "请问",
            "讲了什么",
            "主要内容",
            "总结一下",
            "概述一下",
            "是什么",
            "有哪些",
            "多少",
            "哪些",
            "什么",
            "吗",
            "呢",
            "？",
            "?",
        ):
            cleaned = cleaned.replace(marker, " ")
        tokens = [token.strip() for token in re.findall(r"[\u4e00-\u9fff]{2,8}", cleaned)]
        return [token for token in tokens if token]

    def _meaningful_cjk_chars(self, text: str) -> set[str]:
        return {
            char
            for char in text
            if "\u4e00" <= char <= "\u9fff" and char not in {"的", "了", "是", "这", "那", "有", "和", "吗", "呢", "什么"}
        }

    def _is_summary_question(self, question: str) -> bool:
        normalized = question.lower()
        return any(
            marker in normalized
            for marker in (
                "讲了什么",
                "主要内容",
                "总结",
                "概述",
                "概要",
                "介绍了什么",
                "what is this document about",
                "what does this document say",
                "summarize",
                "summary",
                "overview",
            )
        )

    def _trim_sentence_end(self, text: str, *, language: str) -> str:
        end_chars = "。！？；.!?;"
        trimmed = text.strip()
        while trimmed and trimmed[-1] in end_chars:
            trimmed = trimmed[:-1].rstrip()
        if not trimmed:
            return "证据不足" if language == "zh" else "insufficient evidence"
        return trimmed

    def _ensure_sentence_end(self, text: str, *, language: str) -> str:
        trimmed = text.strip()
        if not trimmed:
            return ""
        if trimmed[-1] in "。！？；.!?;":
            return trimmed
        return f"{trimmed}{'。' if language == 'zh' else '.'}"
