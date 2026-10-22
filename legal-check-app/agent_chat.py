"""
agent_chat.py — Chatbot Agent
Responsible for: Document Q&A and What-If scenario simulation.
Called by the orchestrator in agent.py.

LLM backend: uses llm_adapter (Groq if configured, else Z-AI GLM-4-plus).
"""
from llm_adapter import chat as _call_llm
from prompts import get_system_prompt, qa_prompt, scenario_prompt


def _call(system: str, user: str, temperature: float = 0.2) -> str:
    return _call_llm(system, user, temperature=temperature, max_tokens=4096)


def run_qa(document_text: str, question: str, lang: str = "en") -> str:
    """
    Answer a question grounded strictly in the document.
    Temperature 0.1 — factual, no hallucination.
    """
    return _call(get_system_prompt(lang), qa_prompt(document_text, question, lang), temperature=0.1)


def run_scenario(document_text: str, scenario: str, lang: str = "en") -> str:
    """
    What-if scenario simulation: impact, financial/legal consequences,
    risk level, and advice — all grounded in the document.
    """
    return _call(get_system_prompt(lang), scenario_prompt(document_text, scenario, lang), temperature=0.2)


# ── Suggested starter questions shown in the chatbot UI ──
SCENARIO_SUGGESTIONS_EN = [
    "What if I miss a payment?",
    "What if I want to cancel early?",
    "What if there's a dispute?",
    "What if the other party breaches?",
    "What if I sublease without permission?",
    "What if I'm late for 30 days?",
    "What are my key obligations?",
    "Are there any hidden fees?",
]

SCENARIO_SUGGESTIONS_HI = [
    "अगर मैं भुगतान चूक जाऊं तो?",
    "अगर मैं जल्दी रद्द करना चाहूं तो?",
    "अगर विवाद हो जाए तो?",
    "अगर दूसरा पक्ष उल्लंघन करे तो?",
    "अगर मैं बिना अनुमति के उप-पट्टा करूं तो?",
    "अगर मैं 30 दिन देरी से भुगतान करूं तो?",
    "मेरे मुख्य दायित्व क्या हैं?",
    "क्या कोई छिपे हुए शुल्क हैं?",
]

# backward-compatible default
SCENARIO_SUGGESTIONS = SCENARIO_SUGGESTIONS_EN


def get_scenario_suggestions(lang: str = "en") -> list:
    return SCENARIO_SUGGESTIONS_HI if lang == "hi" else SCENARIO_SUGGESTIONS_EN
