"""
agent_analysis.py — Analysis Agent
Responsible for: Full document analysis + all 8 advanced modules.
Called by the orchestrator in agent.py. Never import app.py.

LLM backend: uses llm_adapter (Groq openai/gpt-oss-20b if GROQ_API_KEY is set,
else Z-AI GLM-4-plus fallback so the app runs in any environment).
"""
import time
from llm_adapter import chat as _call_llm
from prompts import (
    get_system_prompt,
    analysis_prompt,
    ghost_negotiator_prompt,
    heatmap_bias_prompt,
    future_risk_simulator_prompt,
    risk_timeline_prompt,
    smart_negotiation_prompt,
    kfs_audit_prompt,
    prepayment_foreclosure_prompt,
    lc_discrepancy_prompt,
    co_lending_radar_prompt,
)


def _call(system: str, user: str, temperature: float = 0.2, max_tokens: int = 4096) -> str:
    return _call_llm(system, user, temperature=temperature, max_tokens=max_tokens)


def run_core_analysis(document_text: str, lang: str = "en") -> str:
    """Core risk/fraud/obligations/verdict analysis."""
    return _call(get_system_prompt(lang), analysis_prompt(document_text, lang))


def run_ghost_negotiator(document_text: str, risky_clause: str, lang: str = "en") -> str:
    """3 counter-clause versions + send-ready messages."""
    return _call(get_system_prompt(lang), ghost_negotiator_prompt(document_text, risky_clause, lang), temperature=0.4)


def run_heatmap_bias(document_text: str, lang: str = "en") -> str:
    return _call(get_system_prompt(lang), heatmap_bias_prompt(document_text, lang))


def run_future_risk(document_text: str, lang: str = "en") -> str:
    return _call(get_system_prompt(lang), future_risk_simulator_prompt(document_text, lang), temperature=0.3)


def run_risk_timeline(document_text: str, lang: str = "en") -> str:
    return _call(get_system_prompt(lang), risk_timeline_prompt(document_text, lang))


def run_smart_negotiation(document_text: str, lang: str = "en") -> str:
    return _call(get_system_prompt(lang), smart_negotiation_prompt(document_text, lang), temperature=0.3)


def run_kfs_audit(document_text: str, lang: str = "en") -> str:
    return _call(get_system_prompt(lang), kfs_audit_prompt(document_text, lang), temperature=0.1)


def run_prepayment(document_text: str, lang: str = "en") -> str:
    return _call(get_system_prompt(lang), prepayment_foreclosure_prompt(document_text, lang), temperature=0.1)


def run_lc_discrepancy(document_text: str, lang: str = "en") -> str:
    return _call(get_system_prompt(lang), lc_discrepancy_prompt(document_text, lang), temperature=0.1)


def run_co_lending(document_text: str, lang: str = "en") -> str:
    return _call(get_system_prompt(lang), co_lending_radar_prompt(document_text, lang))


# ── Module registry — drives the "Run All" sequential pipeline ──
ADVANCED_MODULES = [
    ("heatmap",      "🔥 Heatmap Bias Flag",               run_heatmap_bias),
    ("future_risk",  "🔮 Future Risk Simulator",           run_future_risk),
    ("timeline",     "📈 Risk Timeline",                    run_risk_timeline),
    ("negotiation",  "🤝 Smart Negotiation",               run_smart_negotiation),
    ("kfs",          "🧾 KFS Integrity Audit",             run_kfs_audit),
    ("prepayment",   "🏦 Pre-Payment & Foreclosure",       run_prepayment),
    ("lc",           "📑 LC Discrepancy Detector",         run_lc_discrepancy),
    ("colending",    "⚖️ Co-Lending Radar",                run_co_lending),
]


def run_all_advanced(document_text: str, progress_callback=None, pause: float = 3.0, lang: str = "en") -> dict:
    """
    Run all 8 advanced modules sequentially with rate-limit pausing.
    progress_callback(step, total, label) is optional.
    Returns dict keyed by module key.
    """
    results = {}
    total = len(ADVANCED_MODULES)
    for i, (key, label, fn) in enumerate(ADVANCED_MODULES):
        if progress_callback:
            progress_callback(i, total, label)
        try:
            results[key] = fn(document_text, lang)
        except Exception as e:
            results[key] = f"⚠️ Module error: {e}"
        if i < total - 1:
            time.sleep(pause)
    if progress_callback:
        progress_callback(total, total, "✅ Done")
    return results
