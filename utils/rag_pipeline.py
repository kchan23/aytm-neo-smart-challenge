"""RAG pipeline: document loading, chunking, and lightweight retrieval for building codes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    text: str
    source: str          # filename
    chunk_index: int
    metadata: dict = field(default_factory=dict)  # page number, line range, etc.


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class BuildingCodeRetriever:
    """
    Load building code documents from a directory, chunk them, build an index,
    and retrieve the top-k most relevant chunks for a given query.

    Retrieval method:
      - "tfidf"     (default) — sklearn TfidfVectorizer, no extra downloads
      - "embedding" — sentence-transformers all-MiniLM-L6-v2, semantic search
    """

    def __init__(
        self,
        data_dir: Path,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        method: str = "tfidf",
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method

        self._chunks: list[DocumentChunk] = []
        self._index = None          # TF-IDF matrix or numpy embedding matrix
        self._vectorizer = None     # TfidfVectorizer (tfidf mode only)
        self._embed_model = None    # SentenceTransformer (embedding mode only)
        self._built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_documents(self) -> list[DocumentChunk]:
        """Load and chunk all supported files from data_dir."""
        chunks: list[DocumentChunk] = []
        supported = {".pdf", ".txt", ".md"}

        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() not in supported:
                continue
            try:
                if path.suffix.lower() == ".pdf":
                    texts = self._load_pdf(path)
                else:
                    texts = self._load_text(path)

                for page_idx, text in enumerate(texts):
                    new_chunks = self._chunk_text(text, path.name, page_idx)
                    chunks.extend(new_chunks)
            except Exception as e:
                print(f"[rag_pipeline] Warning: could not load {path.name}: {e}")

        self._chunks = chunks
        return chunks

    def build_index(self) -> None:
        """Build the retrieval index over loaded chunks."""
        if not self._chunks:
            self._built = True
            return

        texts = [c.text for c in self._chunks]

        if self.method == "embedding":
            self._build_embedding_index(texts)
        else:
            self._build_tfidf_index(texts)

        self._built = True

    def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Return top_k most relevant chunks for query."""
        if not self._built:
            self.build_index()

        if not self._chunks:
            return []

        if self.method == "embedding":
            return self._retrieve_embedding(query, top_k)
        else:
            return self._retrieve_tfidf(query, top_k)

    def format_context(self, chunks: list[DocumentChunk], max_chars: int = 3000) -> str:
        """Join chunks into a single context string with source attribution."""
        parts = []
        total = 0
        for chunk in chunks:
            header = f"[Source: {chunk.source}, chunk {chunk.chunk_index}]"
            block = f"{header}\n{chunk.text}"
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > len(header) + 20:
                    block = block[:remaining] + "…"
                    parts.append(block)
                break
            parts.append(block)
            total += len(block) + 2  # +2 for \n\n separator

        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    @property
    def doc_names(self) -> list[str]:
        seen = []
        for c in self._chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def _load_pdf(self, path: Path) -> list[str]:
        """Load a PDF and return a list of page text strings."""
        # Try PyMuPDF first
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            return [page.get_text() for page in doc]
        except ImportError:
            pass

        # Fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                return [page.extract_text() or "" for page in pdf.pages]
        except ImportError:
            pass

        raise ImportError(
            "No PDF library available. Install PyMuPDF: pip install pymupdf\n"
            "Or pdfplumber: pip install pdfplumber"
        )

    def _load_text(self, path: Path) -> list[str]:
        """Load a plain text or markdown file, split into paragraphs."""
        content = path.read_text(encoding="utf-8", errors="replace")
        # Split on double newlines (paragraph boundaries)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
        # Group paragraphs into ~page-sized blocks for uniform chunking
        block_size = 10
        blocks = []
        for i in range(0, len(paragraphs), block_size):
            blocks.append("\n\n".join(paragraphs[i : i + block_size]))
        return blocks if blocks else [content]

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_text(
        self, text: str, source: str, page_idx: int = 0
    ) -> list[DocumentChunk]:
        """Sliding-window word chunking preserving sentence boundaries."""
        if not text.strip():
            return []

        # Split into sentences (simple heuristic)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        words: list[str] = []
        sentence_boundaries: list[int] = []  # word index where each sentence starts

        for sent in sentences:
            sentence_boundaries.append(len(words))
            words.extend(sent.split())

        if not words:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))

            # Extend end to a sentence boundary if possible
            next_boundary = next(
                (b for b in sentence_boundaries if b > end), None
            )
            if next_boundary and next_boundary - end < 30:
                end = next_boundary

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_idx,
                    metadata={"page": page_idx, "word_start": start, "word_end": end},
                )
            )

            chunk_idx += 1
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    # ------------------------------------------------------------------
    # TF-IDF index
    # ------------------------------------------------------------------

    def _build_tfidf_index(self, texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10_000,
            ngram_range=(1, 2),
        )
        self._index = self._vectorizer.fit_transform(texts)

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[DocumentChunk]:
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._index).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self._chunks[i] for i in top_indices if scores[i] > 0]

    # ------------------------------------------------------------------
    # Embedding index
    # ------------------------------------------------------------------

    def _build_embedding_index(self, texts: list[str]) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers\n"
                "Or switch retrieval method to 'tfidf'."
            )
        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._index = self._embed_model.encode(texts, show_progress_bar=False)

    def _retrieve_embedding(self, query: str, top_k: int) -> list[DocumentChunk]:
        query_vec = self._embed_model.encode([query])
        # Cosine similarity: normalize then dot product
        norm_index = self._index / (
            np.linalg.norm(self._index, axis=1, keepdims=True) + 1e-10
        )
        norm_query = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        scores = norm_index @ norm_query.T
        scores = scores.flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self._chunks[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Streamlit cache helper
# ---------------------------------------------------------------------------

def get_retriever(data_dir: Path, method: str = "tfidf") -> BuildingCodeRetriever:
    """
    Build and return a retriever. Wrap with @st.cache_resource in the page file.
    Separate function so Streamlit can hash the arguments.
    """
    retriever = BuildingCodeRetriever(data_dir, method=method)
    retriever.load_documents()
    retriever.build_index()
    return retriever
