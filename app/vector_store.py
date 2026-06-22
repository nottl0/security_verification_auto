import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt"}

@dataclass
class _Settings:
    knowledge_base_dir: Path = field(
        default_factory=lambda: Path(os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base"))
    )
    vector_store_dir: Path = field(
        default_factory=lambda: Path(os.getenv("VECTOR_STORE_DIR", "./.chroma"))
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "soc_knowledge_base")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
    )
    chunk_size_chars: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE_CHARS", "1000"))
    )
    chunk_overlap_chars: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "3"))
    )
    rebuild: bool = field(
        default_factory=lambda: os.getenv("REBUILD_VECTOR_STORE", "true").lower() == "true"
    )

settings = _Settings()

@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_id: str
    score: float

# Returns a list of (source_identifier, full_text, extension) for dynamic files
def _read_documents(kb_dir: Path) -> list[tuple[str, str, str]]:
    dynamic_dir = kb_dir / "dynamic"
    
    if not dynamic_dir.exists():
        logger.warning(f"Dynamic knowledge base directory not found at: {dynamic_dir}")
        return []

    docs = []
    for path in sorted(dynamic_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
            source_identifier = f"dynamic/{path.name}"
            docs.append((source_identifier, text, path.suffix.lower()))

    if not docs:
        logger.warning(f"No documents found to index inside: {dynamic_dir}")

    return docs

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def _chunk_text(text: str, extension: str) -> list[tuple[str, dict]]:
    if '##' in text:
        # Split by markdown headers to get structural components
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=True
        )
        md_header_docs = markdown_splitter.split_text(text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size_chars,
            chunk_overlap=settings.chunk_overlap_chars,
            separators=["\n\n", "\n", " ", ""]
        )
        
        refined_docs = text_splitter.split_documents(md_header_docs)
        
        final_chunks = []
        for doc in refined_docs:
            if not doc.page_content.strip():
                continue
                
            header_prefix = ""
            if "Header_1" in doc.metadata:
                header_prefix += f"# {doc.metadata['Header_1']}\n"
            if "Header_2" in doc.metadata:
                header_prefix += f"## {doc.metadata['Header_2']}\n"
                
            full_chunk_text = f"{header_prefix}\n{doc.page_content}".strip()
            final_chunks.append((full_chunk_text, doc.metadata))
            
        return final_chunks
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    chunk_size = settings.chunk_size_chars
    overlap = settings.chunk_overlap_chars

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append((current, {}))
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{para}".strip()
        else:
            for i in range(0, len(para), chunk_size - overlap):
                chunks.append((para[i : i + chunk_size], {}))
            current = ""
    if current:
        chunks.append((current, {}))
        
    return chunks

def _make_chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    clean_source = source.replace("/", "_")
    return f"{clean_source}::chunk{index}::{digest}"

class VectorStore:
    def __init__(self) -> None:
        self._client = None
        self._embedding_fn = None
        self._collection = None

    def _ensure_client(self) -> None:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(settings.vector_store_dir))
        if self._embedding_fn is None:
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
    
    # Chunk, embed and build the vector store
    def build(self, kb_dir: Path | None = None, rebuild: bool = True) -> int:
        kb_dir = kb_dir or settings.knowledge_base_dir
        self._ensure_client()

        if rebuild:
            try:
                self._client.delete_collection(settings.collection_name)
            except Exception:
                pass

        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        if not rebuild and self._collection.count() > 0:
            logger.info("Vector store already populated (%d chunks); skipping rebuild", self._collection.count())
            return self._collection.count()

        documents = _read_documents(kb_dir)
        if not documents:
            logger.warning("No dynamic playbooks to ingest")
            return 0

        ids, texts, metadatas = [], [], []
        for source, full_text, ext in documents:
            chunks = _chunk_text(full_text, ext)
            for i, (chunk, header_meta) in enumerate(chunks):
                ids.append(_make_chunk_id(source, i, chunk))
                texts.append(chunk)
                
                base_metadata = {"source": source, "chunk_index": i}
                combined_metadata = {**base_metadata, **header_meta}
                metadatas.append(combined_metadata)

        if not ids:
            raise ValueError("Chunking produced zero chunks; check the dynamic folder")

        self._collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logger.info("Vector store built cleanly: %d dynamic documents -> %d chunks", len(documents), len(ids))
        return len(ids)

    # Searching the vector database with quiery 
    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if self._collection is None:
            raise RuntimeError("Vector store has not been built yet. Call build() first")

        top_k = top_k or settings.top_k
        result = self._collection.query(query_texts=[query], n_results=top_k)

        chunks: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for _id, doc, meta, dist in zip(ids, docs, metas, distances):
            similarity = max(0.0, 1.0 - dist)
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    chunk_id=_id,
                    score=round(similarity, 4),
                )
            )
        return chunks

vector_store = VectorStore()

def build_vector_store() -> int:
    return vector_store.build(rebuild=settings.rebuild)

def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    return vector_store.search(query, top_k=top_k)