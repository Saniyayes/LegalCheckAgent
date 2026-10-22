"""
agent_formfill.py — Form Filler Agent
Responsible for: Analysing uploaded forms and generating a precise
"How to Fill This Form" guide. Only reports fields with direct evidence.
Called by the orchestrator in agent.py. Never import app.py.

LLM backend: uses llm_adapter (Groq if configured, else Z-AI GLM-4-plus).
"""
from llm_adapter import chat as _call_llm
from prompts import get_system_prompt, formfill_prompt


def _call(system: str, user: str, temperature: float = 0.1, max_tokens: int = 4096) -> str:
    return _call_llm(system, user, temperature=temperature, max_tokens=max_tokens)


def run_formfill_analysis(document_text: str, lang: str = "en") -> str:
    """
    Analyse a form document and return a structured 'How to Fill' guide.
    Strict: only returns fields with direct evidence quotes from the document.
    """
    return _call(
        get_system_prompt(lang),
        formfill_prompt(document_text, lang),
        temperature=0.1,
        max_tokens=4096,
    )
