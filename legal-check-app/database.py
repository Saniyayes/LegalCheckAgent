"""
database.py — SQLite Persistence Layer (SQLAlchemy ORM)

Stores everything the LegalCheck app produces, so analyses / chat history / RAG
chunks survive across Streamlit reruns and process restarts.

Schema:
  documents        — uploaded source documents (raw text + metadata)
  analyses         — one analysis result per (document, module) pair
  chat_sessions     — chat session metadata (which doc, which mode)
  chat_messages     — individual user/assistant turns within a session
  graph_runs       — one row per LangGraph orchestration run (state + trace)
  rag_chunks       — metadata for each chunk stored in the Chroma vector store

This module is intentionally side-effect-free at import time. The engine is
created lazily on first use so that importing `database` from many modules
remains cheap.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine, Column, String, Integer, Text, DateTime, ForeignKey, Index,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

# ─────────────────────────────────────────────────────────────────────────────
# Engine setup
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = Path(os.environ.get("LEGALCHECK_DB_DIR", BASE_DIR / "db"))
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "legalcheck.db"
DB_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False because Streamlit calls us from many threads.
_engine = create_engine(DB_URL, echo=False, future=True,
                        connect_args={"check_same_thread": False})
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False,
                               autoflush=False)
Base = declarative_base()


@contextmanager
def session_scope() -> Session:
    """Transactional scope around a series of operations."""
    s = _SessionFactory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    Base.metadata.create_all(_engine)


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def _uuid() -> str:
    return uuid.uuid4().hex


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)        # "pdf" | "docx" | "txt"
    word_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    language = Column(String, default="en")
    raw_text = Column(Text, nullable=False)
    doc_hash = Column(String, index=True)             # SHA-256 of raw_text for dedup
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    analyses = relationship("Analysis", back_populates="document",
                            cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="document",
                                  cascade="all, delete-orphan")
    rag_chunks = relationship("RagChunk", back_populates="document",
                              cascade="all, delete-orphan")


class Analysis(Base):
    """One row per (document, module_name) pair."""
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    module = Column(String, nullable=False)          # "core", "leverage", "heatmap", ...
    language = Column(String, default="en")
    result_text = Column(Text, nullable=False)
    leverage_score = Column(Integer, nullable=True)  # parsed when module=="leverage"
    duration_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    document = relationship("Document", back_populates="analyses")


class ChatSession(Base):
    """A chat session — either 'qa' (Q&A over a doc) or 'courtroom' (case chat)."""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=True, index=True)
    mode = Column(String, nullable=False)            # "qa" | "courtroom"
    case_key = Column(String, nullable=True)         # only for courtroom mode
    title = Column(String, default="Untitled chat")
    language = Column(String, default="en")
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=dt.datetime.utcnow,
                         onupdate=dt.datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session",
                             cascade="all, delete-orphan",
                             order_by="ChatMessage.seq")
    document = relationship("Document", back_populates="chat_sessions")


class ChatMessage(Base):
    """Individual turn in a chat session."""
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"),
                         nullable=False, index=True)
    seq = Column(Integer, default=0)                  # monotonic per-session order
    role = Column(String, nullable=False)             # "user" | "assistant"
    content = Column(Text, nullable=False)
    retrieved_chunks = Column(Text, nullable=True)    # JSON list of chunk_ids used (RAG)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class GraphRun(Base):
    """One LangGraph orchestration run — state + node trace + final output."""
    __tablename__ = "graph_runs"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=True, index=True)
    run_label = Column(String, default="full_pipeline")
    status = Column(String, default="running")        # running|completed|error
    state_json = Column(Text, nullable=True)          # final StateGraph state snapshot
    trace_json = Column(Text, nullable=True)          # list of {"node","started","ended","ok"}
    started_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
    ended_at = Column(DateTime, nullable=True)


class RagChunk(Base):
    """Metadata mirror of every chunk stored in ChromaDB (text lives in Chroma)."""
    __tablename__ = "rag_chunks"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"),
                          nullable=False, index=True)
    chroma_id = Column(String, nullable=False, index=True)  # Chroma collection doc id
    chunk_index = Column(Integer, default=0)
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    document = relationship("Document", back_populates="rag_chunks")


# ─────────────────────────────────────────────────────────────────────────────
# CRUD helpers (used by RAG, LangGraph, and Streamlit UI)
# ─────────────────────────────────────────────────────────────────────────────

def upsert_document(*, filename: str, file_type: str, raw_text: str,
                    language: str = "en", doc_hash: Optional[str] = None) -> Document:
    """Insert a new document row and return it."""
    with session_scope() as s:
        doc = Document(
            filename=filename,
            file_type=file_type,
            raw_text=raw_text,
            char_count=len(raw_text),
            word_count=len(raw_text.split()),
            language=language,
            doc_hash=doc_hash,
        )
        s.add(doc)
        s.flush()
        s.refresh(doc)
        # Detach so caller can read attributes after session close.
        s.expunge(doc)
        return doc


def save_analysis(*, document_id: str, module: str, result_text: str,
                  language: str = "en", leverage_score: Optional[int] = None,
                  duration_ms: int = 0) -> Analysis:
    with session_scope() as s:
        a = Analysis(
            document_id=document_id,
            module=module,
            language=language,
            result_text=result_text,
            leverage_score=leverage_score,
            duration_ms=duration_ms,
        )
        s.add(a)
        s.flush()
        s.refresh(a)
        s.expunge(a)
        return a


def list_documents(limit: int = 50) -> list[Document]:
    with session_scope() as s:
        rows = (s.query(Document)
                 .order_by(Document.created_at.desc())
                 .limit(limit)
                 .all())
        for r in rows:
            s.expunge(r)
        return rows


def get_document(doc_id: str) -> Optional[Document]:
    with session_scope() as s:
        d = s.get(Document, doc_id)
        if d:
            s.expunge(d)
        return d


def list_analyses(document_id: str) -> list[Analysis]:
    with session_scope() as s:
        rows = (s.query(Analysis)
                 .filter(Analysis.document_id == document_id)
                 .order_by(Analysis.created_at.desc())
                 .all())
        for r in rows:
            s.expunge(r)
        return rows


# ── Chat session helpers ───────────────────────────────────────────────────

def create_chat_session(*, document_id: Optional[str], mode: str,
                        case_key: Optional[str] = None,
                        title: str = "Untitled chat",
                        language: str = "en") -> ChatSession:
    with session_scope() as s:
        sess = ChatSession(
            document_id=document_id, mode=mode,
            case_key=case_key, title=title, language=language,
        )
        s.add(sess)
        s.flush()
        s.refresh(sess)
        s.expunge(sess)
        return sess


def append_message(*, session_id: str, role: str, content: str,
                   retrieved_chunks: Optional[list] = None) -> ChatMessage:
    with session_scope() as s:
        seq = (s.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .count())
        msg = ChatMessage(
            session_id=session_id,
            seq=seq,
            role=role,
            content=content,
            retrieved_chunks=json.dumps(retrieved_chunks) if retrieved_chunks else None,
        )
        s.add(msg)
        # touch session.updated_at
        sess = s.get(ChatSession, session_id)
        if sess:
            sess.updated_at = dt.datetime.now(dt.timezone.utc)
        s.flush()
        s.refresh(msg)
        s.expunge(msg)
        return msg


def list_messages(session_id: str) -> list[ChatMessage]:
    with session_scope() as s:
        rows = (s.query(ChatMessage)
                 .filter(ChatMessage.session_id == session_id)
                 .order_by(ChatMessage.seq.asc())
                 .all())
        for r in rows:
            s.expunge(r)
        return rows


def list_chat_sessions(mode: Optional[str] = None, limit: int = 30) -> list[ChatSession]:
    with session_scope() as s:
        q = s.query(ChatSession)
        if mode:
            q = q.filter(ChatSession.mode == mode)
        rows = q.order_by(ChatSession.updated_at.desc()).limit(limit).all()
        for r in rows:
            s.expunge(r)
        return rows


# ── GraphRun helpers ──────────────────────────────────────────────────────

def start_graph_run(*, document_id: Optional[str], run_label: str = "full_pipeline") -> GraphRun:
    with session_scope() as s:
        g = GraphRun(document_id=document_id, run_label=run_label,
                     status="running")
        s.add(g)
        s.flush()
        s.refresh(g)
        s.expunge(g)
        return g


def finish_graph_run(run_id: str, *, status: str, state: dict, trace: list) -> None:
    with session_scope() as s:
        g = s.get(GraphRun, run_id)
        if not g:
            return
        g.status = status
        g.state_json = json.dumps(state, ensure_ascii=False, default=str)
        g.trace_json = json.dumps(trace, ensure_ascii=False, default=str)
        g.ended_at = dt.datetime.now(dt.timezone.utc)


def list_graph_runs(limit: int = 20) -> list[GraphRun]:
    with session_scope() as s:
        rows = (s.query(GraphRun)
                 .order_by(GraphRun.started_at.desc())
                 .limit(limit)
                 .all())
        for r in rows:
            s.expunge(r)
        return rows


# ── RagChunk helpers ──────────────────────────────────────────────────────

def add_rag_chunks(*, document_id: str, chunks: list[dict]) -> int:
    """Bulk-insert RAG chunk metadata. Each chunk: {chroma_id, chunk_index, text, tokens}."""
    with session_scope() as s:
        for c in chunks:
            s.add(RagChunk(
                document_id=document_id,
                chroma_id=c["chroma_id"],
                chunk_index=c.get("chunk_index", 0),
                chunk_text=c["text"],
                token_count=c.get("tokens", 0),
            ))
        return len(chunks)


def delete_rag_chunks(document_id: str) -> int:
    with session_scope() as s:
        n = (s.query(RagChunk)
              .filter(RagChunk.document_id == document_id)
              .delete())
        return n


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

# Initialise tables on first import so downstream modules can rely on schema
# existing. This is idempotent.
init_db()


if __name__ == "__main__":
    # Smoke test
    print(f"Database: {DB_PATH}")
    print(f"Engine URL: {DB_URL}")
    init_db()
    print("Tables created.")

    doc = upsert_document(filename="test.txt", file_type="txt",
                          raw_text="hello world", doc_hash="abc")
    print(f"Inserted doc: {doc.id} ({doc.word_count} words)")
    a = save_analysis(document_id=doc.id, module="core",
                      result_text="ok", duration_ms=42)
    print(f"Inserted analysis: module={a.module} dur={a.duration_ms}ms")
    sess = create_chat_session(document_id=doc.id, mode="qa", title="Test chat")
    print(f"Created chat session: {sess.id}")
    append_message(session_id=sess.id, role="user", content="hi")
    append_message(session_id=sess.id, role="assistant", content="hello")
    msgs = list_messages(sess.id)
    print(f"Messages: {[(m.role, m.content) for m in msgs]}")

    print(f"\nDocuments: {len(list_documents())}")
    print(f"Chat sessions: {len(list_chat_sessions())}")
    print("DB smoke test passed.")
