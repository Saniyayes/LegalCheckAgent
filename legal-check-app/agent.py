"""
agent.py — Orchestrator
Thin coordinator that delegates to specialist agents.
All existing function signatures preserved so nothing downstream breaks.

Specialist agents:
  agent_analysis.py   -> core analysis + 8 advanced modules
  agent_leverage.py   -> leverage radar
  agent_chat.py       -> Q&A + scenario simulation
  agent_courtroom.py  -> AI Courtroom defence attorney chatbot
"""

from agent_analysis import (
    run_core_analysis as run_full_analysis,
    run_ghost_negotiator,
    run_heatmap_bias,
    run_future_risk,
    run_risk_timeline,
    run_smart_negotiation,
    run_kfs_audit,
    run_prepayment,
    run_lc_discrepancy,
    run_co_lending,
    run_all_advanced,
    ADVANCED_MODULES,
)

from agent_leverage import (
    run_leverage_mapping,
    extract_leverage_score,
)

from agent_chat import (
    run_qa,
    run_scenario,
    SCENARIO_SUGGESTIONS,
    get_scenario_suggestions,
)

from agent_courtroom import (
    run_courtroom_chat,
    COURTROOM_CASES,
    CASE_KEYS,
    get_case_field,
)

from agent_formfill import (
    run_formfill_analysis,
)

__all__ = [
    # Analysis
    "run_full_analysis",
    "run_ghost_negotiator",
    "run_heatmap_bias",
    "run_future_risk",
    "run_risk_timeline",
    "run_smart_negotiation",
    "run_kfs_audit",
    "run_prepayment",
    "run_lc_discrepancy",
    "run_co_lending",
    "run_all_advanced",
    "ADVANCED_MODULES",
    # Leverage
    "run_leverage_mapping",
    "extract_leverage_score",
    # Chat
    "run_qa",
    "run_scenario",
    "SCENARIO_SUGGESTIONS",
    "get_scenario_suggestions",
    # Courtroom
    "run_courtroom_chat",
    "COURTROOM_CASES",
    "CASE_KEYS",
    "get_case_field",
    # Form Filler
    "run_formfill_analysis",
]