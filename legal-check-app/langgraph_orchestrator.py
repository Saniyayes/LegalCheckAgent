"""
langgraph_orchestrator.py — LangGraph-powered multi-agent orchestration

This is the centerpiece of the new architecture. It builds a StateGraph that
chains the existing specialist agents (analysis, leverage, advanced modules)
plus a RAG-grounded Q&A node, with conditional edges based on the leverage
score.

Topology:

   START
     ↓
   ingest_doc  →  builds Chroma vector store for RAG, persists to DB
     ↓
   core_analysis  →  calls agent_analysis.run_core_analysis
     ↓
   leverage      →  calls agent_leverage.run_leverage_mapping
     ↓
   <conditional>
     ├── leverage_score >= 60  →  future_risk  (high-confidence path)
     ├── 40 <= score < 60      →  negotiation  (medium-leverage path)
     └── score < 40            →  ghost_negotiator (weak-position path)
     ↓
   heatmap_bias  →  runs bias detection
     ↓
   rag_qa        →  answers a pre-set "summary question" via RAG
     ↓
   finalize      →  persists final state to DB, marks GraphRun as completed
     ↓
   END

Every node:
  • Updates the shared State (TypedDict below).
  • Appends a trace entry {node, started, ended, ok, error}.
  • Persists its result into the SQLite `analyses` table.

The graph is compiled once on import and re-invoked per run via run_pipeline().
"""
from __future__ import annotations

import datetime as dt
import time
import traceback
from typing import TypedDict, Optional, Any, Callable

from langgraph.graph import StateGraph, END

import database
import rag_engine
from agent_analysis import (
    run_core_analysis, run_ghost_negotiator, run_heatmap_bias,
    run_future_risk, run_smart_negotiation,
)
from agent_leverage import run_leverage_mapping, extract_leverage_score as _parse_leverage


# ─────────────────────────────────────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    # Inputs
    document_id: str
    document_text: str
    language: str

    # Control flow
    graph_run_id: str
    leverage_score: Optional[int]

    # Outputs from each node
    ingest_info: dict
    core_analysis: str
    leverage_analysis: str
    branch_output: str          # whichever branch node ran
    branch_name: str             # "future_risk" | "negotiation" | "ghost_negotiator"
    heatmap_bias: str
    rag_answer: dict             # {answer, sources, duration_ms}

    # Tracing
    trace: list[dict]
    status: str                  # running | completed | error


# ─────────────────────────────────────────────────────────────────────────────
# Node implementations — each takes state, returns partial state to merge.
# ─────────────────────────────────────────────────────────────────────────────

def _trace_entry(node: str, *, ok: bool, started: float, ended: float,
                  error: Optional[str] = None) -> dict:
    return {
        "node": node,
        "started_at": dt.datetime.fromtimestamp(started,
                                                 tz=dt.timezone.utc).isoformat(),
        "ended_at": dt.datetime.fromtimestamp(ended,
                                               tz=dt.timezone.utc).isoformat(),
        "duration_ms": int((ended - started) * 1000),
        "ok": ok,
        "error": error,
    }


def _save(state: GraphState, module: str, result: str,
          leverage_score: Optional[int] = None, duration_ms: int = 0) -> None:
    """Persist node output to SQLite."""
    try:
        database.save_analysis(
            document_id=state["document_id"],
            module=module,
            result_text=result,
            language=state.get("language", "en"),
            leverage_score=leverage_score,
            duration_ms=duration_ms,
        )
    except Exception:
        # Persistence must never crash the pipeline.
        traceback.print_exc()


# ── Node: ingest_doc ─────────────────────────────────────────────────────────

def node_ingest_doc(state: GraphState) -> dict:
    started = time.time()
    try:
        info = rag_engine.ingest_document(
            document_id=state["document_id"],
            raw_text=state["document_text"],
            source_name=state.get("document_id", "uploaded_document"),
        )
        trace = state.get("trace", []) + [_trace_entry("ingest_doc", ok=True,
                                                         started=started,
                                                         ended=time.time())]
        return {"ingest_info": info, "trace": trace}
    except Exception as e:
        trace = state.get("trace", []) + [_trace_entry("ingest_doc", ok=False,
                                                         started=started,
                                                         ended=time.time(),
                                                         error=str(e))]
        return {"trace": trace, "status": "error"}


# ── Node: core_analysis ────────────────────────────────────────────────────

def node_core_analysis(state: GraphState) -> dict:
    started = time.time()
    lang = state.get("language", "en")
    try:
        result = run_core_analysis(state["document_text"], lang=lang)
        _save(state, "core", result, duration_ms=int((time.time() - started) * 1000))
        trace = state.get("trace", []) + [_trace_entry("core_analysis", ok=True,
                                                         started=started,
                                                         ended=time.time())]
        return {"core_analysis": result, "trace": trace}
    except Exception as e:
        traceback.print_exc()
        trace = state.get("trace", []) + [_trace_entry("core_analysis", ok=False,
                                                         started=started,
                                                         ended=time.time(),
                                                         error=str(e))]
        return {"trace": trace}


# ── Node: leverage ──────────────────────────────────────────────────────────

def node_leverage(state: GraphState) -> dict:
    started = time.time()
    lang = state.get("language", "en")
    try:
        result = run_leverage_mapping(state["document_text"], lang=lang)
        score = _parse_leverage(result)
        _save(state, "leverage", result, leverage_score=score,
              duration_ms=int((time.time() - started) * 1000))
        trace = state.get("trace", []) + [_trace_entry("leverage", ok=True,
                                                         started=started,
                                                         ended=time.time())]
        return {"leverage_analysis": result,
                "leverage_score": score,
                "trace": trace}
    except Exception as e:
        traceback.print_exc()
        trace = state.get("trace", []) + [_trace_entry("leverage", ok=False,
                                                         started=started,
                                                         ended=time.time(),
                                                         error=str(e))]
        return {"trace": trace}


# ── Conditional edge: pick the branch based on leverage_score ─────────────────

def route_by_leverage(state: GraphState) -> str:
    """Return next node name based on leverage score."""
    score = state.get("leverage_score")
    if score is None:
        return "ghost_negotiator"     # safe default for weak/unknown position
    if score >= 60:
        return "future_risk"
    if score >= 40:
        return "negotiation"
    return "ghost_negotiator"


# ── Branch nodes ────────────────────────────────────────────────────────────

def node_future_risk(state: GraphState) -> dict:
    started = time.time()
    lang = state.get("language", "en")
    try:
        result = run_future_risk(state["document_text"], lang=lang)
        _save(state, "future_risk", result,
              duration_ms=int((time.time() - started) * 1000))
        trace = state.get("trace", []) + [_trace_entry("future_risk", ok=True,
                                                         started=started,
                                                         ended=time.time())]
        return {"branch_output": result, "branch_name": "future_risk", "trace": trace}
    except Exception as e:
        traceback.print_exc()
        trace = state.get("trace", []) + [_trace_entry("future_risk", ok=False,
                                                         started=started,
                                                         ended=time.time(),
                                                         error=str(e))]
        return {"trace": trace}


def node_smart_negotiation(state: GraphState) -> dict:
    started = time.time()
    lang = state.get("language", "en")
    try:
        result = run_smart_negotiation(state["document_text"], lang=lang)
        _save(state, "smart_negotiation", result,
              duration_ms=int((time.time() - started) * 1000))
        trace = state.get("trace", []) + [_trace_entry("smart_negotiation", ok=True,
                                                         started=started,
                                                         ended=time.time())]
        return {"branch_output": result, "branch_name": "negotiation", "trace": trace}
    except Exception as e:
        traceback.print_exc()
        trace = state.get("trace", []) + [_trace_entry("smart_negotiation", ok=False,
                                                         started=started,
                                                         ended=time.time(),
                                                         error=str(e))]
        return {"trace": trace}


def node_ghost_negotiator(state: GraphState) -> dict:
    """For weak leverage positions: ask LLM to draft counter-clauses generically."""
    started = time.time()
    lang = state.get("language", "en")
    try:
        # Without a specific risky_clause, we feed the whole doc as the risky_clause.
        result = run_ghost_negotiator(state["document_text"],
                                      state["document_text"][:2000],
                                      lang=lang)
        _save(state, "ghost_negotiator", result,
              duration_ms=int((time.time() - started) * 1000))
        trace = state.get("trace", []) + [_trace_entry("ghost_negotiator", ok=True,
                                                         started=started,
                                                         ended=time.time())]
        return {"branch_output": result, "branch_name": "ghost_negotiator", "trace": trace}
    except Exception as e:
        traceback.print_exc()
        trace = state.get("trace", []) + [_trace_entry("ghost_negotiator", ok=False,
                                                         started=started,
                                                         ended=time.time(),
                                                         error=str(e))]
        return {"trace": trace}


# ── Node: heatmap_bias ──────────────────────────────────────────────────────

def node_heatmap_bias(state: GraphState) -> dict:
    started = time.time()
    lang = state.get("language", "en")
    try:
        result = run_heatmap_bias(state["document_text"], lang=lang)
        _save(state, "heatmap_bias", result,
              duration_ms=int((time.time() - started) * 1000))
        trace = state.get("trace", []) + [_trace_entry("heatmap_bias", ok=True,
                                                         started=started,
                                                         ended=time.time())]
        return {"heatmap_bias": result, "trace": trace}
    except Exception as e:
        traceback.print_exc()
        trace = state.get("trace", []) + [_trace_entry("heatmap_bias", ok=False,
                                                         started=started,
                                                         ended=time.time(),
                                                         error=str(e))]
        return {"trace": trace}


# ── Node: rag_qa ────────────────────────────────────────────────────────────

DEFAULT_RAG_QUESTION = (
    "What are the three most important clauses the signer should review before signing?"
)


def node_rag_qa(state: GraphState) -> dict:
    started = time.time()
    try:
        res = rag_engine.answer_with_rag(
            document_id=state["document_id"],
            question=DEFAULT_RAG_QUESTION,
            k=4,
            temperature=0.1,
        )
        trace = state.get("trace", []) + [_trace_entry("rag_qa", ok=True,
                                                         started=started,
                                                         ended=time.time())]
        return {"rag_answer": res, "trace": trace}
    except Exception as e:
        traceback.print_exc()
        trace = state.get("trace", []) + [_trace_entry("rag_qa", ok=False,
                                                         started=started,
                                                         ended=time.time(),
                                                         error=str(e))]
        return {"trace": trace}


# ── Node: finalize ──────────────────────────────────────────────────────────

def node_finalize(state: GraphState) -> dict:
    """Mark the GraphRun row as completed."""
    started = time.time()
    try:
        trace = state.get("trace", []) + [_trace_entry("finalize", ok=True,
                                                         started=started,
                                                         ended=time.time())]
        if state.get("graph_run_id"):
            database.finish_graph_run(
                state["graph_run_id"],
                status="completed",
                state=_serialize_state(state),
                trace=trace,
            )
        return {"trace": trace, "status": "completed"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error"}


def _serialize_state(state: GraphState) -> dict:
    """Strip non-serializable bits before persisting."""
    safe_keys = [
        "document_id", "language", "leverage_score", "branch_name",
        "core_analysis", "leverage_analysis", "branch_output",
        "heatmap_bias", "ingest_info", "status",
    ]
    out = {}
    for k in safe_keys:
        if k in state:
            v = state[k]
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                out[k] = v
    # rag_answer: keep answer + source chunk_ids only
    if "rag_answer" in state:
        ra = state["rag_answer"] or {}
        out["rag_answer"] = {
            "answer": ra.get("answer", ""),
            "sources": [c.get("chunk_id") for c in ra.get("sources", [])],
            "duration_ms": ra.get("duration_ms"),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("ingest_doc",      node_ingest_doc)
    g.add_node("core_analysis",   node_core_analysis)
    g.add_node("leverage",        node_leverage)
    g.add_node("future_risk",     node_future_risk)
    g.add_node("negotiation",     node_smart_negotiation)
    g.add_node("ghost_negotiator", node_ghost_negotiator)
    g.add_node("heatmap_bias",    node_heatmap_bias)
    g.add_node("rag_qa",          node_rag_qa)
    g.add_node("finalize",        node_finalize)

    g.set_entry_point("ingest_doc")
    g.add_edge("ingest_doc",     "core_analysis")
    g.add_edge("core_analysis",  "leverage")

    # Conditional branch after leverage
    g.add_conditional_edges(
        "leverage",
        route_by_leverage,
        {
            "future_risk":      "future_risk",
            "negotiation":      "negotiation",
            "ghost_negotiator": "ghost_negotiator",
        },
    )

    # All three branches converge into heatmap_bias
    g.add_edge("future_risk",      "heatmap_bias")
    g.add_edge("negotiation",      "heatmap_bias")
    g.add_edge("ghost_negotiator", "heatmap_bias")

    g.add_edge("heatmap_bias", "rag_qa")
    g.add_edge("rag_qa",      "finalize")
    g.add_edge("finalize",    END)

    return g.compile()


# Compile once at import time. If compilation fails, the error surfaces
# immediately rather than mid-pipeline.
COMPILED_GRAPH = build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(document_id: str, document_text: str,
                 language: str = "en",
                 on_step: Optional[Callable[[str, dict], None]] = None) -> dict:
    """
    Run the full LangGraph pipeline on a document.

    Args:
        document_id:    row id in the `documents` table
        document_text:  raw text extracted from the uploaded file
        language:       "en" or "hi"
        on_step:        optional callback(step_name, partial_state) — for live UI
    Returns:
        The final GraphState dict (with all node outputs + trace).
    """
    # Create a GraphRun row
    gr = database.start_graph_run(document_id=document_id,
                                   run_label="langgraph_full_pipeline")

    initial_state: GraphState = {
        "document_id": document_id,
        "document_text": document_text,
        "language": language,
        "graph_run_id": gr.id,
        "trace": [],
        "status": "running",
    }

    # Use stream() so we can call on_step after each node.
    final_state = initial_state
    try:
        for event in COMPILED_GRAPH.stream(initial_state, {"recursion_limit": 30}):
            # event is {node_name: partial_state}
            for node_name, partial in event.items():
                if on_step:
                    try:
                        on_step(node_name, partial)
                    except Exception:
                        pass
                # merge into our running snapshot (already merged by LangGraph)
                if isinstance(partial, dict):
                    final_state.update(partial)
    except Exception as e:
        traceback.print_exc()
        final_state["status"] = "error"
        if gr.id:
            database.finish_graph_run(
                gr.id, status="error",
                state={"error": str(e)},
                trace=final_state.get("trace", []),
            )
        return final_state

    # Finalize already runs inside the graph; ensure DB row is in sync.
    if final_state.get("status") != "completed":
        database.finish_graph_run(
            gr.id, status=final_state.get("status", "completed"),
            state=_serialize_state(final_state),
            trace=final_state.get("trace", []),
        )
    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import database

    sample = """
    RESIDENTIAL LEASE AGREEMENT

    This Lease Agreement is entered into on 1st January 2025 between Mr. Anil Kapoor
    (Landlord) and Ms. Priya Sharma (Tenant).

    1. PREMISES: The Landlord rents Apartment 4B, Hill View Apartments, Mumbai to the Tenant.

    2. TERM: 11 months commencing 1st January 2025.

    3. RENT: Rs. 35,000 per month, payable by 5th of each month.

    4. SECURITY DEPOSIT: Rs. 70,000 refundable within 30 days of vacation.

    5. TERMINATION: 30 days written notice. Landlord may terminate immediately for
       non-payment over 15 days, illegal activities, or structural damage.

    6. DISPUTES: Arbitration under Arbitration and Conciliation Act 1996, seat Mumbai.

    7. GOVERNING LAW: Laws of India.

    8. NO SUBLETTING without prior written consent.

    9. MAINTENANCE: Minor repairs under Rs. 1,000 borne by Tenant.
    """

    doc = database.upsert_document(
        filename="lease_test.txt", file_type="txt",
        raw_text=sample,
        doc_hash=hashlib.sha256(sample.encode()).hexdigest() if (hashlib := __import__("hashlib")) else None,
    )
    print(f"Doc: {doc.id}")

    def progress(node, partial):
        print(f"  ✓ node finished: {node}")

    print("\nRunning LangGraph pipeline...")
    t0 = time.time()
    final = run_pipeline(doc.id, sample, language="en", on_step=progress)
    print(f"\nPipeline completed in {time.time()-t0:.1f}s")
    print(f"Status: {final.get('status')}")
    print(f"Trace nodes: {len(final.get('trace', []))}")
    for t in final.get("trace", []):
        print(f"  - {t['node']:20s} ok={t['ok']}  {t['duration_ms']}ms")

    if final.get("leverage_score") is not None:
        print(f"\nLeverage score: {final['leverage_score']}")
    print(f"Branch taken: {final.get('branch_name', '(none)')}")
    print(f"\nCore analysis (first 200 chars):\n{(final.get('core_analysis') or '')[:200]}")
    print(f"\nRAG answer (first 200 chars):\n{(final.get('rag_answer') or {}).get('answer', '')[:200]}")

    # Verify DB persistence
    print("\n\nDB rows for this document:")
    analyses = database.list_analyses(doc.id)
    print(f"  analyses: {len(analyses)} rows")
    for a in analyses:
        print(f"    [{a.module}] {a.duration_ms}ms  score={a.leverage_score}")
    runs = database.list_graph_runs(limit=5)
    print(f"  graph_runs: {len(runs)} rows (latest: {runs[0].status if runs else 'none'})")
