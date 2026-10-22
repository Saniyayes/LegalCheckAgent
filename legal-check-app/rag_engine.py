"""
rag_engine.py — Retrieval-Augmented Generation with LangChain + ChromaDB

Pipeline:
  1. Split raw document text into chunks (RecursiveCharacterTextSplitter).
  2. Embed each chunk with HuggingFace all-MiniLM-L6-v2 (runs locally, no API key).
  3. Persist embeddings in a per-document Chroma collection.
  4. Mirror chunk metadata into SQLite (database.RagChunk) for SQL-level browsing.
  5. Expose a retriever + answer_with_rag() helper that grounds LLM responses in
     the top-k retrieved chunks and returns the source chunk IDs alongside the
     answer.

The retriever is reused by the LangGraph orchestrator and by the Streamlit
chat UI.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_DIR = Path(os.environ.get("LEGALCHECK_CHROMA_DIR", Path(__file__).resolve().parent / "chroma_db"))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000           # chars
CHUNK_OVERLAP = 200         # chars
DEFAULT_K = 4               # chunks to retrieve per query


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton: embeddings + persistent Chroma client
# (HuggingFace model loads ~300MB and takes ~5s on first import, so we lazy-load)
# ─────────────────────────────────────────────────────────────────────────────

_embeddings = None
_chroma_client = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        # langchain_huggingface is the actively maintained package.
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


# ─────────────────────────────────────────────────────────────────────────────
# Document ingestion: chunk + embed + persist
# ─────────────────────────────────────────────────────────────────────────────

def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_document(document_id: str, raw_text: str,
                    source_name: str = "uploaded_document") -> dict:
    """
    Chunk and embed a document into a dedicated Chroma collection.

    Returns a summary dict: {chunks_added, collection_name, duration_ms}.
    Idempotent: if the collection already exists for this document, it is
    recreated from scratch (avoids duplicate IDs).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document as LCDocument
    from langchain_chroma import Chroma

    t0 = time.time()
    collection_name = f"doc_{document_id[:12]}"

    # 1. Drop existing collection for this doc (idempotent re-ingest).
    try:
        _get_chroma_client().delete_collection(collection_name)
    except Exception:
        pass  # collection didn't exist — fine.

    # 2. Clear RagChunk rows for this doc in SQLite.
    import database
    database.delete_rag_chunks(document_id)

    # 3. Chunk the text.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)
    if not chunks:
        return {"chunks_added": 0, "collection_name": collection_name,
                "duration_ms": int((time.time() - t0) * 1000)}

    # 4. Build LangChain Document objects with metadata.
    chunk_docs = []
    chunk_meta = []
    for i, ctext in enumerate(chunks):
        cid = f"{document_id}_c{i:04d}"
        chunk_docs.append(LCDocument(
            page_content=ctext,
            metadata={
                "chunk_id": cid,
                "chunk_index": i,
                "document_id": document_id,
                "source": source_name,
            },
        ))
        chunk_meta.append({
            "chroma_id": cid,
            "chunk_index": i,
            "text": ctext,
            "tokens": len(ctext.split()),
        })

    # 5. Embed + persist into Chroma.
    vectordb = Chroma.from_documents(
        documents=chunk_docs,
        embedding=_get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )

    # 6. Mirror metadata into SQLite.
    database.add_rag_chunks(document_id=document_id, chunks=chunk_meta)

    return {
        "chunks_added": len(chunks),
        "collection_name": collection_name,
        "duration_ms": int((time.time() - t0) * 1000),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def get_retriever(document_id: str, k: int = DEFAULT_K):
    """Return a LangChain retriever for the document's Chroma collection."""
    from langchain_chroma import Chroma
    collection_name = f"doc_{document_id[:12]}"
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve(document_id: str, query: str, k: int = DEFAULT_K) -> list[dict]:
    """
    Retrieve top-k chunks for a query.

    Returns a list of {chunk_id, text, score, chunk_index, source}.
    """
    from langchain_chroma import Chroma
    collection_name = f"doc_{document_id[:12]}"
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    results = vectordb.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in results:
        md = doc.metadata or {}
        out.append({
            "chunk_id": md.get("chunk_id", ""),
            "chunk_index": md.get("chunk_index", 0),
            "text": doc.page_content,
            "score": float(score),
            "source": md.get("source", ""),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# RAG-grounded answering
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are LegalCheck's Retrieval-Augmented assistant.

You answer the user's question STRICTLY grounded in the retrieved document
excerpts provided below. Hard rules:

  • If the answer is supported by an excerpt, cite the chunk_id like [chunk_3].
  • If the excerpts do not contain the answer, say:
        "I could not find this in the uploaded document."
    Do NOT speculate, paraphrase from memory, or invent clauses.
  • Keep the answer concise (3-6 sentences) and direct.
  • When asked about risks, obligations, or hidden clauses, prefer the most
    specific excerpt that mentions the term.

Retrieved excerpts:
{context}
"""


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no chunks retrieved)"
    lines = []
    for c in chunks:
        lines.append(f"[{c['chunk_id']}] (chunk #{c['chunk_index']}, score={c['score']:.3f})")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines).strip()


def answer_with_rag(document_id: str, question: str,
                    k: int = DEFAULT_K,
                    temperature: float = 0.1) -> dict:
    """
    Retrieve chunks for `question`, then ask the LLM to answer using only
    those chunks.

    Returns: {answer, sources (list of chunk dicts), duration_ms}.
    """
    import llm_adapter
    t0 = time.time()
    chunks = retrieve(document_id, question, k=k)
    context = _format_context(chunks)
    system = RAG_SYSTEM_PROMPT.format(context=context)
    answer = llm_adapter.chat(system, question,
                              temperature=temperature, max_tokens=2048)
    return {
        "answer": answer,
        "sources": chunks,
        "duration_ms": int((time.time() - t0) * 1000),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Maintenance helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_collections() -> list[str]:
    return [c.name for c in _get_chroma_client().list_collections()]


def drop_document_vectors(document_id: str) -> bool:
    collection_name = f"doc_{document_id[:12]}"
    try:
        _get_chroma_client().delete_collection(collection_name)
        import database
        database.delete_rag_chunks(document_id)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import database

    sample = """
    RESIDENTIAL LEASE AGREEMENT

    This Lease Agreement is entered into on 1st January 2025 between Mr. Anil Kapoor
    (hereinafter "Landlord") and Ms. Priya Sharma (hereinafter "Tenant").

    1. PREMISES: The Landlord agrees to rent Apartment 4B, Hill View Apartments,
       Mumbai 400001 to the Tenant for residential use only.

    2. TERM: The lease shall be for a period of 11 (eleven) months commencing
       1st January 2025 and ending 30th November 2025.

    3. RENT: The Tenant shall pay a monthly rent of Rs. 35,000 (Rupees Thirty-Five
       Thousand only) on or before the 5th day of each calendar month.

    4. SECURITY DEPOSIT: The Tenant has paid a refundable security deposit of
       Rs. 70,000 (two months' rent). This deposit shall be refunded within 30 days
       of vacation of premises, after deducting any unpaid dues.

    5. TERMINATION: Either party may terminate this agreement with 30 days' written
       notice. The Landlord may terminate immediately in case of:
       (a) non-payment of rent for 15 consecutive days,
       (b) illegal activities on the premises,
       (c) structural damage caused by the Tenant.

    6. DISPUTE RESOLUTION: Any disputes shall be resolved by arbitration under the
       Arbitration and Conciliation Act 1996. The seat of arbitration shall be Mumbai.

    7. GOVERNING LAW: This agreement shall be governed by the laws of India.

    8. NO SUBLETTING: The Tenant shall not sublet or assign the premises without
       prior written consent of the Landlord.

    9. MAINTENANCE: Minor repairs (under Rs. 1,000) shall be borne by the Tenant.
       Major repairs shall be the Landlord's responsibility.
    """

    doc = database.upsert_document(
        filename="lease_sample.txt",
        file_type="txt",
        raw_text=sample,
        doc_hash=_doc_hash(sample),
    )
    print(f"Document inserted: {doc.id}")

    info = ingest_document(doc.id, sample, source_name="lease_sample.txt")
    print(f"Ingested: {info}")

    print("\nRetrieval test — query: 'What is the rent?'")
    chunks = retrieve(doc.id, "What is the monthly rent?", k=3)
    for c in chunks:
        print(f"  [{c['chunk_id']}] score={c['score']:.3f}: {c['text'][:120]!r}")

    print("\nRAG answer test:")
    res = answer_with_rag(doc.id, "When must the tenant pay rent each month?")
    print(f"Answer ({res['duration_ms']}ms):\n{res['answer']}")
    print(f"Sources: {[c['chunk_id'] for c in res['sources']]}")
