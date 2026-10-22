"""
llm_adapter.py — Unified LLM Provider
Provides a single chat() interface that any module in this project can call.

Provider selection:
  1. Groq (openai/gpt-oss-20b)           — used when GROQ_API_KEY env var is set.
  2. Z-AI CLI / GLM-4-plus (fallback)    — used automatically when Groq is not configured,
                                          so the application runs out-of-the-box in this
                                          sandbox. Calls the `z-ai chat` CLI under the hood.

Both providers accept the same (system, user, temperature, max_tokens) signature
and return a plain string — exactly what every existing agent_*.py module expects.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Provider detection
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_provider() -> str:
    """Return the active LLM provider: 'groq' or 'zai'."""
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if shutil.which("z-ai"):
        return "zai"
    # Last-resort deterministic stub so the UI still boots for demos.
    return "stub"


def provider_label() -> str:
    """Human-readable provider label for the Streamlit sidebar."""
    p = get_provider()
    if p == "groq":
        return f"Groq · {GROQ_MODEL}"
    if p == "zai":
        return "Z-AI · GLM-4-plus (fallback)"
    return "Stub · deterministic (no API key configured)"


# ──────────────────────────────────────────────────────────────────────────────
# Groq provider
# ──────────────────────────────────────────────────────────────────────────────

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _call_groq(system: str, user: str, temperature: float, max_tokens: int) -> str:
    client = _groq_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        extra_body={"reasoning_effort": "low"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def _call_groq_with_history(system: str, history: list[dict],
                            temperature: float, max_tokens: int) -> str:
    client = _groq_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        extra_body={"reasoning_effort": "low"},
        messages=[{"role": "system", "content": system}] + history,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────────────────────────────────────────
# Z-AI CLI provider (GLM-4-plus) — fallback
# ──────────────────────────────────────────────────────────────────────────────

_ZAI_LOCK = threading.Lock()  # serialize CLI calls (avoids race on temp files)


def _strip_cli_banner(text: str) -> str:
    """Drop the '🚀 Initializing...' / '🚀 Sending...' lines from z-ai output."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if line.startswith("🚀"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _call_zai(system: str, user: str, temperature: float, max_tokens: int) -> str:
    """
    Invoke the z-ai CLI to perform a chat completion.
    The CLI accepts --system and --prompt and emits a JSON body.
    """
    cmd = [
        "z-ai", "chat",
        "--system", system,
        "--prompt", user,
    ]
    with _ZAI_LOCK:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"z-ai CLI failed: {proc.stderr[:500]}")
    cleaned = _strip_cli_banner(proc.stdout)
    try:
        body = json.loads(cleaned)
        return body["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        # Sometimes the CLI prints trailing non-JSON lines; try to salvage the
        # first {...} block that looks like a chat.completion object.
        match = re.search(r"\{[\s\S]*\"choices\"[\s\S]*\}", cleaned)
        if match:
            try:
                body = json.loads(match.group(0))
                return body["choices"][0]["message"]["content"]
            except Exception:
                pass
        raise RuntimeError(
            f"z-ai CLI returned non-JSON output. First 400 chars: {cleaned[:400]}"
        ) from e


def _call_zai_with_history(system: str, history: list[dict],
                           temperature: float, max_tokens: int) -> str:
    """
    Emulate multi-turn chat by flattening history into a single user prompt.
    GLM-4-plus handles this well — we just prepend prior turns as context.
    """
    lines = [f"[System Instructions]\n{system}\n"]
    for turn in history:
        role = turn.get("role", "user").upper()
        lines.append(f"[{role}]\n{turn.get('content', '')}\n")
    lines.append("[Assistant]\n")
    flat_prompt = "\n".join(lines)
    return _call_zai("", flat_prompt, temperature, max_tokens)


# ──────────────────────────────────────────────────────────────────────────────
# Stub provider (last resort — no API key, no z-ai CLI)
# ──────────────────────────────────────────────────────────────────────────────

def _call_stub(system: str, user: str, temperature: float, max_tokens: int) -> str:
    snippet = (user[:300] + "…") if len(user) > 300 else user
    return (
        "⚠️ **No LLM provider configured.**\n\n"
        "Set `GROQ_API_KEY` in your environment, or run inside this sandbox "
        "where the `z-ai` CLI provides GLM-4-plus as a fallback.\n\n"
        f"_Preview of input ({len(user)} chars):_\n\n```\n{snippet}\n```"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def chat(system: str, user: str,
         temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """Single-turn chat completion. Returns assistant text."""
    provider = get_provider()
    if provider == "groq":
        return _call_groq(system, user, temperature, max_tokens)
    if provider == "zai":
        return _call_zai(system, user, temperature, max_tokens)
    return _call_stub(system, user, temperature, max_tokens)


def chat_with_history(system: str, history: list[dict],
                      temperature: float = 0.35, max_tokens: int = 4096) -> str:
    """Multi-turn chat completion. `history` is a list of {role, content} dicts."""
    provider = get_provider()
    if provider == "groq":
        return _call_groq_with_history(system, history, temperature, max_tokens)
    if provider == "zai":
        return _call_zai_with_history(system, history, temperature, max_tokens)
    last_user = next((t["content"] for t in reversed(history)
                      if t.get("role") == "user"), "")
    return _call_stub(system, last_user, temperature, max_tokens)


if __name__ == "__main__":
    # Quick smoke test
    print(f"Active provider: {get_provider()}  ({provider_label()})")
    t0 = time.time()
    out = chat(
        "You are a helpful assistant. Answer in one short sentence.",
        "What is 7 × 8?",
        temperature=0.0,
    )
    print(f"Response ({time.time()-t0:.1f}s):\n{out}")
