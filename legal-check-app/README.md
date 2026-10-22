# LegalCheck.AI-Agent — LangGraph + RAG + DB Persistence Edition

This is an enhanced version of the original [LegalCheck.AI-Agent](https://github.com/Saniyayes/ai_legal_agent) Streamlit app, augmented with three production-grade capabilities on top of the existing 5-agent legal document analysis system:

1. **LangGraph orchestration** — a typed `StateGraph` that chains the existing specialist agents into a single multi-agent pipeline with a conditional branch based on the leverage score.
2. **LangChain RAG** — a retrieval-augmented-generation layer (Chroma vector store + HuggingFace `all-MiniLM-L6-v2` embeddings) that grounds chat answers strictly in retrieved document chunks and cites source `chunk_id`s.
3. **SQLite persistence** — every uploaded document, every analysis output, every chat turn, every LangGraph run (with full state + trace) is persisted to a SQLite database so history survives across reruns and restarts.

The app runs **out-of-the-box in any environment** thanks to an LLM-provider adapter that uses Groq (`openai/gpt-oss-20b`) when `GROQ_API_KEY` is set, and falls back automatically to GLM-4-plus via the `z-ai` CLI when the key is absent.

---

## Architecture

```
                              ┌──────────────────────┐
                              │   Streamlit UI       │
                              │   (app.py)           │
                              └──────────┬───────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
   ┌────────────────────┐    ┌─────────────────────┐    ┌────────────────────┐
   │  Original 5 tabs   │    │  🧠 LangGraph + RAG │    │  🗄️ History & DB   │
   │  (auto-pipeline)   │    │  (run pipeline +    │    │  (browse persisted │
   │                    │    │   RAG chat)          │    │   data)            │
   └─────────┬──────────┘    └──────────┬──────────┘    └─────────┬──────────┘
             │                            │                        │
             │   uses                     │   uses                  │   reads
             ▼                            ▼                        │
   ┌────────────────────┐    ┌──────────────────────────┐         │
   │  agent_*.py        │    │  langgraph_orchestrator  │         │
   │  (analysis,        │    │  (StateGraph builder)    │         │
   │   leverage, chat,  │    │                          │         │
   │   courtroom,       │    │  rag_engine              │         │
   │   formfill)        │    │  (Chroma + HF embeds)    │         │
   └─────────┬──────────┘    └──────────┬───────────────┘         │
             │                            │                         │
             └────────────┬───────────────┘                         │
                          ▼                                         │
                ┌─────────────────────┐                              │
                │   llm_adapter       │  Groq ↔ Z-AI CLI (GLM-4)    │
                └─────────────────────┘                              │
                          │                                          │
                          ▼                                          │
                ┌──────────────────────────────────────────────────┘
                │
                ▼
        ┌────────────────┐
        │  database.py   │  SQLAlchemy + SQLite
        │                │  documents / analyses / chat_sessions /
        │                │  chat_messages / graph_runs / rag_chunks
        └────────────────┘
```

### LangGraph topology

```
  START
    │
    ▼
  ingest_doc   ──►  chunks → Chroma vector store + SQLite mirror
    │
    ▼
  core_analysis   (risk / fraud / obligations / verdict)
    │
    ▼
  leverage   ──►  parse leverage_score (0–100)
    │
    ▼   (conditional edge)
    ├── score ≥ 60      ──► future_risk
    ├── 40 ≤ score < 60 ──► smart_negotiation
    └── score < 40      ──► ghost_negotiator
    │
    ▼
  heatmap_bias   (clause bias detection)
    │
    ▼
  rag_qa   ──►  retrieve top-4 chunks → LLM grounded answer
    │
    ▼
  finalize   ──►  persist full state to graph_runs table
    │
    ▼
   END
```

Every node:
- Updates the shared `GraphState` (a `TypedDict`).
- Appends a trace entry `{node, started, ended, ok, error, duration_ms}`.
- Persists its output to the SQLite `analyses` table (with the parsed leverage score where applicable).

The final state and full trace are persisted as JSON in the `graph_runs` table.

---

## Database schema

All tables live in `/home/z/my-project/db/legalcheck.db` (SQLite).

| Table | Purpose |
|---|---|
| `documents` | One row per uploaded file. Dedup by SHA-256 content hash. |
| `analyses` | One row per `(document, module)` pair. Captures module name, language, result text, leverage_score (when applicable), duration_ms. |
| `chat_sessions` | Chat session metadata: mode (`qa` / `qa_rag` / `courtroom`), case_key, language, title. |
| `chat_messages` | Individual turns within a session. Includes optional `retrieved_chunks` JSON for RAG messages. |
| `graph_runs` | One row per LangGraph pipeline run: status, full state JSON, full trace JSON, started_at, ended_at. |
| `rag_chunks` | Mirror of every chunk in ChromaDB: `chroma_id`, `chunk_index`, `chunk_text`, `token_count`. Lets the UI show chunk contents without re-reading Chroma. |

---

## Files added / modified

| File | Status | Purpose |
|---|---|---|
| `llm_adapter.py` | **new** | Unified LLM interface: Groq if `GROQ_API_KEY` is set, else GLM-4-plus via `z-ai` CLI. |
| `database.py` | **new** | SQLAlchemy ORM models + CRUD helpers for all 6 tables. |
| `rag_engine.py` | **new** | Chroma vector store, `RecursiveCharacterTextSplitter`, HuggingFace embeddings, `answer_with_rag()`. |
| `langgraph_orchestrator.py` | **new** | `StateGraph` definition + node implementations + `run_pipeline()`. |
| `app.py` | **modified** | Added tabs 6 & 7, wired sidebar to show provider + DB status, hooked upload + chat to DB persistence. |
| `agent_analysis.py` | **modified** | Replaced direct Groq client with `llm_adapter.chat()`. Same public API. |
| `agent_chat.py` | **modified** | Same — uses `llm_adapter.chat()`. |
| `agent_leverage.py` | **modified** | Same — uses `llm_adapter.chat()`. |
| `agent_courtroom.py` | **modified** | Same — uses `llm_adapter.chat_with_history()`. |
| `agent_formfill.py` | **modified** | Same — uses `llm_adapter.chat()`. |
| `requirements.txt` | **modified** | Added langchain, langgraph, chromadb, sqlalchemy, sentence-transformers, langchain-huggingface, langchain-chroma, langchain-text-splitters. |
| `scripts/streamlit_launcher.py` | **new** | Robust launcher that fully detaches Streamlit from the parent shell (uses `start_new_session=True`). |
| `sample_lease.txt` | **new** | Sample lease agreement used for end-to-end verification. |

---

## How to run

### Local

```bash
# 1. Install deps (Python 3.10+)
pip install -r requirements.txt

# 2. (Optional) Set Groq API key — otherwise the app falls back to GLM-4-plus
export GROQ_API_KEY="gsk_..."

# 3. Launch Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### In this sandbox

The app is already running on port 8765. To restart it:

```bash
python /home/z/my-project/legal-check-app/scripts/streamlit_launcher.py start
python /home/z/my-project/legal-check-app/scripts/streamlit_launcher.py status
python /home/z/my-project/legal-check-app/scripts/streamlit_launcher.py stop
```

---

## Verification

End-to-end test on `sample_lease.txt` (538 words):

| Node | Duration | OK |
|---|---|---|
| ingest_doc | 14.1s | ✅ |
| core_analysis | 6.8s | ✅ |
| leverage | 7.0s | ✅ |
| ghost_negotiator (branch taken, leverage score = 30) | 7.1s | ✅ |
| heatmap_bias | 5.1s | ✅ |
| rag_qa | 1.6s | ✅ |
| finalize | 0ms | ✅ |
| **Total** | **49.6s** | |

The RAG answer correctly identified the top 3 clauses (PREMISES, RENT, TERMINATION) with chunk citations.

---

## What's new for end users

### 🧠 LangGraph + RAG tab (new)

- Visual pipeline topology diagram (ASCII).
- "🚀 Run LangGraph Pipeline" button — streams node completion in real time.
- After completion: trace table (node / ok / duration_ms / error), branch taken, leverage score, all node outputs in expanders, RAG answer with source `chunk_id`s.
- Interactive RAG chat: each answer cites the retrieved chunks, with a collapsible "Retrieved chunks" panel showing chunk_id, similarity score, and a text preview.
- All chat turns are persisted to the `chat_messages` table with their retrieved chunk IDs.

### 🗄️ History & DB tab (new)

- **Uploaded Documents** table — every document ever uploaded with filename, type, word count, language, upload time.
- Click "🔁 Reload into app" to restore any past document into the live session.
- Click "🔍 View analyses" to see every analysis row ever run on a document.
- Click "🧩 View RAG chunks" to inspect the chunked text stored in the vector store.
- **Graph Runs** table — every LangGraph run with status, doc id, started_at, ended_at. Inspect any run to see its full trace table and final state JSON.
- **Saved Chat Sessions** table — every chat session (Q&A, RAG, courtroom). Inspect any session to replay the full message history with retrieved chunks.

### Sidebar additions

The sidebar now shows:
- 🔌 **LLM**: the active provider (`Groq · openai/gpt-oss-20b` or `Z-AI · GLM-4-plus (fallback)`).
- 🗄️ **DB**: live count of persisted documents.

Every upload is now auto-deduplicated by SHA-256 content hash — uploading the same file twice will reuse the existing document row.
