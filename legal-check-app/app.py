from dotenv import load_dotenv
load_dotenv()
import io
import time
import streamlit as st
from document_parser import extract_text
from agent import (
    run_full_analysis, run_ghost_negotiator,
    run_heatmap_bias, run_future_risk, run_risk_timeline,
    run_smart_negotiation, run_kfs_audit, run_prepayment,
    run_lc_discrepancy, run_co_lending,
    run_leverage_mapping, extract_leverage_score,
    run_qa, run_scenario,
    get_scenario_suggestions,
    run_courtroom_chat, COURTROOM_CASES, CASE_KEYS,
    get_case_field,
    run_formfill_analysis,
)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

# ── New modules: LangGraph + RAG + Database persistence ────────────────────
import hashlib
import datetime as dt
from llm_adapter import get_provider, provider_label
import database
import rag_engine
import langgraph_orchestrator

st.set_page_config(page_title="Legal Check", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
/* MAIN TITLE */
h1 {
    font-size: 48px !important;   /* 🔧 change size here */
    text-align: center;           /* 🔧 center / left / right */
    margin-top: -20px;            /* 🔧 move UP */
    margin-bottom: 10px;          /* spacing below */
}
/* TAB TITLES */
button[data-baseweb="tab"] {
    font-size: 18px !important;   /* 🔧 tab text size */
    font-weight: 600;
}
/* MAKE TABS FULL WIDTH */
div[data-baseweb="tab-list"] {
    display: flex;
    justify-content: space-between;
}

/* EACH TAB TAKES EQUAL SPACE */
button[data-baseweb="tab"] {
    flex: 1;
    text-align: center;
}
button[data-baseweb="tab"] {
    padding: 12px 0px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────
# UI STRINGS — bilingual label map
# ─────────────────────────────────────────────────────────────
UI = {
    "en": {
        "title": "Legal Check",
        "caption": "Upload any legal document — full autonomous analysis begins immediately after upload.",
        "sidebar_header": "Upload Document",
        "uploader_label": "Choose a file",
        "loaded": "Loaded",
        "word_count": "Word count",
        "powered_by": "Powered by Groq · openai/gpt-oss-20b",
        "agents_caption": "Autonomous · 5 agents: Analysis · Leverage · Chat · Courtroom · Form Filler",
        "all_done": "All agents complete",
        "running": "⏳ Agents running...",
        "tab_analysis": "Full Analysis",
        "tab_leverage": "Leverage Radar",
        "tab_chat": "Chatbot",
        "tab_court": "AI Courtroom",
        "upload_prompt": "Upload a document from the sidebar — analysis starts automatically.",
        "running_prompt": "Analysis is running above — results will appear here when complete.",
        "chat_sub": "Chatbot — Ask the Document",
        "chat_caption": "Ask questions or simulate scenarios. Answers are grounded strictly in the document.  |  Agent: **Chat Agent**",
        "chat_upload": "Upload a document from the sidebar to begin.",
        "chat_tap": "**Tap a suggestion to get started, or type your own question below:**",
        "chat_input": "Ask anything about the document, or describe a what-if scenario...",
        "clear_chat": "🗑️ Clear chat",
        "leverage_sub": "Leverage Radar — Power Balance Analysis",
        "leverage_caption": "Agent: **Leverage Agent**",
        "leverage_upload": "Upload a document from the sidebar — analysis starts automatically.",
        "leverage_running": "Leverage analysis is running — results will appear here when complete.",
        "leverage_score_label": "### Your leverage score",
        "leverage_strong": "Strong position",
        "leverage_moderate": "Moderate leverage",
        "leverage_weak": "Weak position — negotiate carefully",
        "leverage_no_result": "Leverage analysis did not return results.",
        "court_sub": "AI Courtroom — Your Personal Defence Attorney",
        "court_caption": (
            "Select a case, read the charges, then chat with your AI defence attorney. "
            "The attorney always argues in your favour.  |  Agent: **Courtroom Agent**"
        ),
        "court_select_case": "#### Select your case",
        "court_selected": "✅ Selected",
        "court_select_btn": "Select Case →",
        "court_pick_prompt": "👆 Select a case above to begin your consultation with the AI defence attorney.",
        "court_quick": "**💡 Quick prompts — tap to start:**",
        "court_thinking": "Your attorney is preparing a response...",
        "court_reset": "🗑️ Reset chat",
        "ghost_expander": "👻 Ghost Negotiator — Fix Risky Clauses",
        "ghost_caption": "Click any flagged clause to generate 3 counter-clause versions + a ready-to-send message.",
        "ghost_fix": "Fix ✨",
        "ghost_copy": "**📋 Copy this message:**",
        "step_core": "🤖 Analysis Agent running core analysis...",
        "step_leverage": "📡 Leverage Agent running... (10/10)",
        "step_done": "✅ All agents complete — results are ready across all tabs.",
        "rate_wait": "⏳ Rate limit hit on {} — waiting 30s...",
        "language_label": "🌐 Language / भाषा",
        "verdict_safe": "### ✅ VERDICT: Safe to Sign",
        "verdict_caution": "### ⚠️ VERDICT: Proceed with Caution",
        "verdict_no": "### ❌ VERDICT: Do NOT Sign",
        "expanders": {
            "📄 Summary":         ("📄 SUMMARY:",         "⚠️ RISKS:"),
            "⚠️ Risks":           ("⚠️ RISKS:",           "📌 OBLIGATIONS:"),
            "📌 Obligations":     ("📌 OBLIGATIONS:",     "🚨 FRAUD CHECK:"),
            "🚨 Fraud Check":     ("🚨 FRAUD CHECK:",     "🔍 TAMPERING CHECK:"),
            "🔍 Tampering":       ("🔍 TAMPERING CHECK:", "📉 MISSING CLAUSES:"),
            "📉 Missing Clauses": ("📉 MISSING CLAUSES:", "🤖 FINAL DECISION:"),
            "🤖 Final Decision":  ("🤖 FINAL DECISION:",  None),
        },
        "adv_expanders": {
            "heatmap":     "🔥 Heatmap Bias Flag",
            "future_risk": "🔮 Future Risk & 📈 Risk Timeline",
            "negotiation": "🤝 Smart Negotiation",
            "kfs":         "🧾 KFS Integrity Audit",
            "prepayment":  "🏦 Pre-Payment & Foreclosure Shield",
            "lc":          "📑 LC Discrepancy Detector",
            "colending":   "⚖️ Co-Lending Radar",
        },
        "raw_output": "Full raw output",
        "tab_formfill": "Form Filler",
        "formfill_sub": "Smart Form Filler — How to Fill Your Form",
        "formfill_caption": "Upload any form (PDF, DOCX, TXT) and get an instant step-by-step guide to fill it correctly.  |  Agent: **Form Filler Agent**",
        "formfill_upload_prompt": "Upload a form document below to get started.",
        "formfill_uploader": "Upload your form here (PDF, DOCX, TXT)",
        "formfill_running": "Form Filler Agent is analysing your form...",
        "formfill_no_result": "The agent could not extract form fields from this document.",
        "formfill_loaded": "Form loaded",
        "formfill_words": "Words detected",
        "formfill_rerun": "Analyse a different form",
        "pdf_download_btn": "⬇Download Analysis as PDF",
        "pdf_generating": "Generating PDF report...",
        "pdf_ready": "✅ PDF ready — click to download!",
        "pdf_report_title": "AI Legal Guardian — Analysis Report",
        # ── New: LangGraph + RAG tab ──
        "tab_langraph": "🧠 LangGraph + RAG",
        "lg_sub": "🧠 LangGraph-Orchestrated Multi-Agent Pipeline + RAG",
        "lg_caption": (
            "Runs the full LangGraph state machine: ingest → core analysis → leverage → "
            "conditional branch (future_risk / negotiation / ghost_negotiator) → heatmap → "
            "RAG-grounded Q&A. Every node persists to SQLite.  |  Agent: **LangGraph Orchestrator**"
        ),
        "lg_upload_prompt": "Upload a document from the sidebar, then click **Run Pipeline**.",
        "lg_run_btn": "🚀 Run LangGraph Pipeline",
        "lg_running": "⏳ Pipeline running… (each node streams below as it finishes)",
        "lg_done": "✅ Pipeline completed — full state snapshot saved to database.",
        "lg_branch_label": "Conditional branch taken",
        "lg_rag_q_label": "RAG-grounded question (auto-asked)",
        "lg_trace_label": "Node execution trace",
        "lg_chat_sub": "💬 RAG Chat — ask the vector store",
        "lg_chat_caption": (
            "Questions answered strictly from retrieved chunks (Chroma + all-MiniLM-L6-v2). "
            "Each answer cites source chunk_ids."
        ),
        "lg_chat_input": "Ask anything — answers cite retrieved chunks…",
        "lg_chat_sources": "**Retrieved chunks:**",
        # ── New: Database / History tab ──
        "tab_db": "🗄️ History & DB",
        "db_sub": "🗄️ Persistence — Documents · Analyses · Chats · Graph Runs",
        "db_caption": (
            "Every upload, analysis, chat turn, and graph run is persisted in SQLite "
            "(`db/legalcheck.db`). Browse, reload, or rerun any past "
            "session here."
        ),
        "db_documents_h": "📄 Uploaded Documents",
        "db_analyses_h": "🤖 Analyses",
        "db_runs_h": "🧠 Graph Runs",
        "db_chats_h": "💬 Saved Chat Sessions",
        "db_no_docs": "No documents persisted yet — upload one to begin.",
        "db_load_btn": "Reload into app",
        "db_view_chunks": "Inspect RAG chunks",
    },
    "hi": {
        "title": "⚖️ AI कानूनी संरक्षक एजेंट",
        "caption": "कोई भी कानूनी दस्तावेज़ अपलोड करें — अपलोड के तुरंत बाद पूर्ण स्वायत्त विश्लेषण शुरू हो जाता है।",
        "sidebar_header": "📁 दस्तावेज़ अपलोड करें",
        "uploader_label": "फ़ाइल चुनें",
        "loaded": "✅ लोड हो गया",
        "word_count": "शब्द गिनती",
        "powered_by": "Groq · openai/gpt-oss-20b द्वारा संचालित",
        "agents_caption": "स्वायत्त · 5 एजेंट: विश्लेषण · लीवरेज · चैट · कोर्टरूम · फ़ॉर्म फिलर",
        "all_done": "✅ सभी एजेंट पूर्ण",
        "running": "⏳ एजेंट चल रहे हैं...",
        "tab_analysis": "📊 पूर्ण विश्लेषण",
        "tab_leverage": "📡 लीवरेज रडार",
        "tab_chat": "💬 चैटबॉट",
        "tab_court": "🏛️ AI कोर्टरूम",
        "upload_prompt": "📁 साइडबार से दस्तावेज़ अपलोड करें — विश्लेषण स्वचालित रूप से शुरू होता है।",
        "running_prompt": "⏳ विश्लेषण चल रहा है — परिणाम पूर्ण होने पर यहां दिखाई देंगे।",
        "chat_sub": "💬 चैटबॉट — दस्तावेज़ से पूछें",
        "chat_caption": "प्रश्न पूछें या परिस्थितियों का अनुकरण करें। उत्तर दस्तावेज़ पर आधारित हैं।  |  एजेंट: **चैट एजेंट**",
        "chat_upload": "📁 शुरू करने के लिए साइडबार से दस्तावेज़ अपलोड करें।",
        "chat_tap": "**💡 शुरू करने के लिए एक सुझाव टैप करें, या नीचे अपना प्रश्न लिखें:**",
        "chat_input": "दस्तावेज़ के बारे में कुछ भी पूछें, या कोई परिस्थिति बताएं...",
        "clear_chat": "🗑️ चैट साफ़ करें",
        "leverage_sub": "📡 लीवरेज रडार — शक्ति संतुलन विश्लेषण",
        "leverage_caption": "एजेंट: **लीवरेज एजेंट**",
        "leverage_upload": "📁 साइडबार से दस्तावेज़ अपलोड करें — विश्लेषण स्वचालित रूप से शुरू होता है।",
        "leverage_running": "⏳ लीवरेज विश्लेषण चल रहा है — परिणाम यहां दिखाई देंगे।",
        "leverage_score_label": "### आपका लीवरेज स्कोर",
        "leverage_strong": "मजबूत स्थिति",
        "leverage_moderate": "मध्यम लीवरेज",
        "leverage_weak": "कमजोर स्थिति — सावधानी से बातचीत करें",
        "leverage_no_result": "लीवरेज विश्लेषण ने परिणाम नहीं दिया।",
        "court_sub": "🏛️ AI कोर्टरूम — आपका व्यक्तिगत बचाव वकील",
        "court_caption": (
            "एक मामला चुनें, आरोप पढ़ें, फिर अपने AI बचाव वकील से बात करें। "
            "वकील हमेशा आपके पक्ष में तर्क देता है।  |  एजेंट: **कोर्टरूम एजेंट**"
        ),
        "court_select_case": "#### अपना मामला चुनें",
        "court_selected": "✅ चुना गया",
        "court_select_btn": "मामला चुनें →",
        "court_pick_prompt": "👆 परामर्श शुरू करने के लिए ऊपर से एक मामला चुनें।",
        "court_quick": "**💡 त्वरित प्रश्न — शुरू करने के लिए टैप करें:**",
        "court_thinking": "आपका वकील उत्तर तैयार कर रहा है...",
        "court_reset": "🗑️ चैट रीसेट करें",
        "ghost_expander": "👻 घोस्ट वार्ताकार — जोखिमपूर्ण धाराएं ठीक करें",
        "ghost_caption": "किसी भी चिह्नित धारा पर क्लिक करें — 3 प्रति-धारा संस्करण + भेजने के लिए तैयार संदेश।",
        "ghost_fix": "ठीक करें ✨",
        "ghost_copy": "**📋 यह संदेश कॉपी करें:**",
        "step_core": "🤖 विश्लेषण एजेंट मुख्य विश्लेषण चला रहा है...",
        "step_leverage": "📡 लीवरेज एजेंट चल रहा है... (10/10)",
        "step_done": "✅ सभी एजेंट पूर्ण — परिणाम सभी टैब में तैयार हैं।",
        "rate_wait": "⏳ {} पर दर सीमा — 30 सेकंड प्रतीक्षा...",
        "language_label": "🌐 Language / भाषा",
        "verdict_safe": "### ✅ फैसला: हस्ताक्षर करना सुरक्षित",
        "verdict_caution": "### ⚠️ फैसला: सावधानी से आगे बढ़ें",
        "verdict_no": "### ❌ फैसला: हस्ताक्षर न करें",
        "expanders": {
            "📄 सारांश":           ("📄 सारांश:",         "⚠️ जोखिम:"),
            "⚠️ जोखिम":            ("⚠️ जोखिम:",          "📌 दायित्व:"),
            "📌 दायित्व":           ("📌 दायित्व:",         "🚨 धोखाधड़ी जांच:"),
            "🚨 धोखाधड़ी जांच":     ("🚨 धोखाधड़ी जांच:",   "🔍 छेड़छाड़ जांच:"),
            "🔍 छेड़छाड़ जांच":      ("🔍 छेड़छाड़ जांच:",    "📉 अनुपस्थित धाराएं:"),
            "📉 अनुपस्थित धाराएं":   ("📉 अनुपस्थित धाराएं:", "🤖 अंतिम निर्णय:"),
            "🤖 अंतिम निर्णय":      ("🤖 अंतिम निर्णय:",    None),
        },
        "adv_expanders": {
            "heatmap":     "🔥 हीटमैप पूर्वाग्रह",
            "future_risk": "🔮 भविष्य जोखिम और 📈 जोखिम टाइमलाइन",
            "negotiation": "🤝 स्मार्ट वार्ता",
            "kfs":         "🧾 KFS ऑडिट",
            "prepayment":  "🏦 पूर्व-भुगतान और फौजदारी",
            "lc":          "📑 LC विसंगति डिटेक्टर",
            "colending":   "⚖️ को-लेंडिंग रडार",
        },
        "raw_output": "🗂️ पूरा कच्चा आउटपुट",
        "tab_formfill": "📝 फ़ॉर्म फिलर",
        "formfill_sub": "📝 स्मार्ट फ़ॉर्म फिलर — फ़ॉर्म कैसे भरें",
        "formfill_caption": "कोई भी फ़ॉर्म (PDF, DOCX, TXT) अपलोड करें और उसे सही तरीके से भरने की तुरंत मार्गदर्शिका प्राप्त करें।  |  एजेंट: **फ़ॉर्म फिलर एजेंट**",
        "formfill_upload_prompt": "📁 शुरू करने के लिए नीचे फ़ॉर्म दस्तावेज़ अपलोड करें।",
        "formfill_uploader": "यहां अपना फ़ॉर्म अपलोड करें (PDF, DOCX, TXT)",
        "formfill_running": "🔍 फ़ॉर्म फिलर एजेंट आपके फ़ॉर्म का विश्लेषण कर रहा है...",
        "formfill_no_result": "एजेंट इस दस्तावेज़ से फ़ॉर्म फ़ील्ड निकालने में असमर्थ रहा।",
        "formfill_loaded": "✅ फ़ॉर्म लोड हुआ",
        "formfill_words": "शब्द मिले",
        "formfill_rerun": "🔄 दूसरा फ़ॉर्म विश्लेषित करें",
        "pdf_download_btn": "⬇️ विश्लेषण PDF डाउनलोड करें",
        "pdf_generating": "PDF रिपोर्ट बना रहे हैं...",
        "pdf_ready": "✅ PDF तैयार है — डाउनलोड करें!",
        "pdf_report_title": "AI कानूनी संरक्षक — विश्लेषण रिपोर्ट",
        # ── New: LangGraph + RAG tab (Hindi) ──
        "tab_langraph": "🧠 LangGraph + RAG",
        "lg_sub": "🧠 LangGraph-ऑर्केस्ट्रेटेड मल्टी-एजेंट पाइपलाइन + RAG",
        "lg_caption": (
            "पूरा LangGraph स्टेट मशीन चलाता है: ingest → कोर विश्लेषण → लीवरेज → "
            "शर्तीय शाखा (future_risk / negotiation / ghost_negotiator) → heatmap → "
            "RAG-आधारित Q&A। हर नोड SQLite में सेव होता है।  |  एजेंट: **LangGraph Orchestrator**"
        ),
        "lg_upload_prompt": "साइडबार से दस्तावेज़ अपलोड करें, फिर **Run Pipeline** पर क्लिक करें।",
        "lg_run_btn": "🚀 LangGraph पाइपलाइन चलाएं",
        "lg_running": "⏳ पाइपलाइन चल रहा है… (हर नोड समाप्त होने पर नीचे दिखेगा)",
        "lg_done": "✅ पाइपलाइन पूर्ण — संपूर्ण स्टेट डेटाबेस में सेव हो गया।",
        "lg_branch_label": "चुनी गई शर्तीय शाखा",
        "lg_rag_q_label": "RAG-आधारित प्रश्न (स्वतः)",
        "lg_trace_label": "नोड एग्जीक्यूशन ट्रेस",
        "lg_chat_sub": "💬 RAG चैट — वेक्टर स्टोर से पूछें",
        "lg_chat_caption": (
            "उत्तर केवल प्राप्त chunks से (Chroma + all-MiniLM-L6-v2)। हर उत्तर source chunk_id उद्धृत करता है।"
        ),
        "lg_chat_input": "कुछ भी पूछें — उत्तर chunks उद्धृत करते हैं…",
        "lg_chat_sources": "**प्राप्त chunks:**",
        # ── New: Database / History tab (Hindi) ──
        "tab_db": "🗄️ हिस्ट्री & DB",
        "db_sub": "🗄️ परसिस्टेंस — दस्तावेज़ · विश्लेषण · चैट्स · ग्राफ रन्स",
        "db_caption": (
            "हर अपलोड, विश्लेषण, चैट टर्न, और ग्राफ रन SQLite में सेव होता है "
            "(`db/legalcheck.db`)। यहां किसी भी पूर्व सत्र को देखें, रीलोड करें या दोबारा चलाएं।"
        ),
        "db_documents_h": "📄 अपलोड किए गए दस्तावेज़",
        "db_analyses_h": "🤖 विश्लेषण",
        "db_runs_h": "🧠 ग्राफ रन्स",
        "db_chats_h": "💬 सेव किए गए चैट सत्र",
        "db_no_docs": "अभी कोई दस्तावेज़ सेव नहीं है — शुरू करने के लिए अपलोड करें।",
        "db_load_btn": "ऐप में रीलोड करें",
        "db_view_chunks": "RAG chunks देखें",
    },
}

# ── Session state ─────────────────────────────────────────────
for k, v in {
    "doc_text": "",
    "auto_done": False,
    "core_analysis": "",
    "adv_results": {},
    "leverage_result": "",
    "ghost_results": {},
    "chat_history": [],
    "scenario_input": "",
    "last_file": "",
    "lang": "en",
    "last_lang": "en",
    "court_case_key": None,
    "court_history": [],
    "court_suggestion_clicked": "",
    "formfill_text": "",
    "formfill_result": "",
    "formfill_last_file": "",
    "analysis_pdf_bytes": None,
    # ── New: LangGraph + RAG + DB persistence ──
    "db_doc_id": None,                # id of the current document row
    "db_doc_hash": None,
    "lg_final_state": None,           # last LangGraph run's final state
    "lg_running": False,
    "lg_progress": [],                # list of trace entries so far
    "rag_chat_history": [],           # messages in the RAG chat tab
    "rag_session_id": None,           # ChatSession row id
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    # Language toggle — first in sidebar
    lang_choice = st.radio(
        UI["en"]["language_label"],
        options=["English", "हिंदी"],
        horizontal=True,
        index=0 if st.session_state.lang == "en" else 1,
    )
    st.session_state.lang = "en" if lang_choice == "English" else "hi"
    L = UI[st.session_state.lang]  # shortcut to current language strings

    # If language changed mid-session, reset analysis so it reruns in new language
    if st.session_state.lang != st.session_state.last_lang:
        st.session_state.auto_done = False
        st.session_state.core_analysis = ""
        st.session_state.adv_results = {}
        st.session_state.leverage_result = ""
        st.session_state.ghost_results = {}
        st.session_state.chat_history = []
        st.session_state.court_history = []
        st.session_state.last_lang = st.session_state.lang

    st.divider()
    st.header(L["sidebar_header"])
    uploaded_file = st.file_uploader(L["uploader_label"], type=["pdf", "docx", "txt"])
    if uploaded_file and uploaded_file.name != st.session_state.last_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        with st.spinner("..."):
            st.session_state.doc_text = extract_text(uploaded_file, ext)
        st.session_state.last_file       = uploaded_file.name
        st.session_state.auto_done       = False
        st.session_state.core_analysis   = ""
        st.session_state.adv_results     = {}
        st.session_state.leverage_result = ""
        st.session_state.ghost_results   = {}
        st.session_state.chat_history    = []
        st.session_state.court_history   = []
        st.session_state.scenario_input  = ""

        # ── Persist upload to SQLite (dedup by content hash) ──────────
        try:
            doc_hash = hashlib.sha256(
                st.session_state.doc_text.encode("utf-8")
            ).hexdigest()
            # Check if a doc with this hash already exists
            from sqlalchemy import text as _sql_text
            with database.session_scope() as _s:
                existing = _s.execute(
                    _sql_text("SELECT id FROM documents WHERE doc_hash = :h LIMIT 1"),
                    {"h": doc_hash},
                ).fetchone()
            if existing:
                st.session_state.db_doc_id = existing[0]
            else:
                doc_row = database.upsert_document(
                    filename=uploaded_file.name,
                    file_type=ext,
                    raw_text=st.session_state.doc_text,
                    language=st.session_state.lang,
                    doc_hash=doc_hash,
                )
                st.session_state.db_doc_id = doc_row.id
            st.session_state.db_doc_hash = doc_hash
        except Exception as e:
            # Persistence must not break the UI
            st.session_state.db_doc_id = None
            print(f"[DB] upload persistence failed: {e}")

        st.success(f"{L['loaded']}: {uploaded_file.name}")
        st.metric(L["word_count"], f"{len(st.session_state.doc_text.split()):,}")
    elif uploaded_file:
        st.success(f"{L['loaded']}: {uploaded_file.name}")
        st.metric(L["word_count"], f"{len(st.session_state.doc_text.split()):,}")

    st.divider()
    st.caption(L["powered_by"])
    st.caption(L["agents_caption"])

    # ── New: show active LLM provider + DB status ─────────────────────
    try:
        prov = provider_label()
        st.caption(f"🔌 LLM: **{prov}**")
    except Exception:
        pass
    try:
        n_docs = len(database.list_documents(limit=1000))
        st.caption(f"🗄️ DB: **{n_docs}** documents persisted")
    except Exception:
        pass

    if st.session_state.auto_done:
        st.success(L["all_done"])
    elif st.session_state.doc_text and not st.session_state.auto_done:
        st.info(L["running"])

lang = st.session_state.lang
L = UI[lang]

st.title(L["title"])
st.caption(L["caption"])
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    L["tab_analysis"], L["tab_leverage"], L["tab_chat"],
    L["tab_court"], L["tab_formfill"],
    L["tab_langraph"], L["tab_db"],
])


# ── Shared helpers ────────────────────────────────────────────
def _render_core(analysis):
    # Verdict detection — works for both languages (emoji anchors)
    if "✅" in analysis and ("Safe to Sign" in analysis or "हस्ताक्षर करना सुरक्षित" in analysis):
        st.success(L["verdict_safe"])
    elif "⚠️" in analysis and ("Proceed with Caution" in analysis or "सावधानी से" in analysis):
        st.warning(L["verdict_caution"])
    elif "❌" in analysis and ("Do NOT Sign" in analysis or "हस्ताक्षर न करें" in analysis):
        st.error(L["verdict_no"])

    for title, (sm, em) in L["expanders"].items():
        if sm not in analysis:
            continue
        s = analysis.find(sm) + len(sm)
        e = analysis.find(em) if em and em in analysis else len(analysis)
        c = analysis[s:e].strip() \
            .replace("🔴 High",   "🔴 **High**") \
            .replace("🟡 Medium", "🟡 **Medium**") \
            .replace("🟢 Low",    "🟢 **Low**") \
            .replace("🔴 उच्च",   "🔴 **उच्च**") \
            .replace("🟡 मध्यम",  "🟡 **मध्यम**") \
            .replace("🟢 कम",     "🟢 **कम**")
        with st.expander(title, expanded=True):
            st.markdown(c)


def _render_ghost(result):
    for sm, em, dtitle, color in [
        ("VERSION 1 — AGGRESSIVE", "VERSION 2 — FAIR",      "🔴 Aggressive — Max protection", "error"),
        ("VERSION 2 — FAIR",       "VERSION 3 — DEFENSIVE",  "🟡 Fair — Industry standard",    "warning"),
        ("VERSION 3 — DEFENSIVE",  "NEGOTIATION TIP:",       "🟢 Defensive — Minimal ask",     "success"),
    ]:
        if sm not in result:
            continue
        s = result.find(sm) + len(sm)
        e = result.find(em) if em in result else len(result)
        content = result[s:e].strip()
        msg = ""
        if "Message to send:" in content and "---" in content:
            ms = content.find("---") + 3
            me = content.rfind("---")
            if me > ms:
                msg = content[ms:me].strip()
        with st.expander(dtitle, expanded=True):
            getattr(st, color)(content)
            if msg:
                st.markdown(L["ghost_copy"])
                st.code(msg, language=None)
    if "NEGOTIATION TIP:" in result:
        st.info(f"💡 {result[result.find('NEGOTIATION TIP:') + len('NEGOTIATION TIP:'):].strip()}")


def _extract_risky_clauses(analysis):
    # Works for both languages — risks section always starts with ⚠️
    marker = "⚠️ RISKS:" if lang == "en" else "⚠️ जोखिम:"
    if marker not in analysis:
        return []
    s = analysis.find(marker) + len(marker)
    e = len(analysis)
    end_markers = ["📌 OBLIGATIONS:", "🚨 FRAUD CHECK:", "📌 दायित्व:", "🚨 धोखाधड़ी जांच:"]
    for m in end_markers:
        if m in analysis:
            e = min(e, analysis.find(m))
    return [{"label": l.strip()[:80], "full": l.strip()}
            for l in analysis[s:e].strip().split("\n") if l.strip().startswith("-")]


def _render_leverage(result):
    score = extract_leverage_score(result)
    if score is not None:
        st.markdown(L["leverage_score_label"])
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            if score >= 60:
                st.success(f"### {score}/100 — {L['leverage_strong']}")
            elif score >= 35:
                st.warning(f"### {score}/100 — {L['leverage_moderate']}")
            else:
                st.error(f"### {score}/100 — {L['leverage_weak']}")
            st.progress(score / 100)
        st.divider()
    for title, (sm, em) in {
        "⚖️ Mutual obligations":               ("MUTUAL OBLIGATIONS",                "ONE-SIDED OBLIGATIONS AGAINST YOU:"),
        "🔴 One-sided obligations against you": ("ONE-SIDED OBLIGATIONS AGAINST YOU:", "ONE-SIDED ADVANTAGES YOU HAVE:"),
        "🟢 Advantages you have":              ("ONE-SIDED ADVANTAGES YOU HAVE:",     "YOUR WALK-AWAY POINTS"),
        "🚶 Walk-away points":                  ("YOUR WALK-AWAY POINTS",              "HIDDEN LEVERAGE"),
        "🔍 Hidden leverage":                   ("HIDDEN LEVERAGE",                    "NEGOTIATION TACTICS"),
        "🎯 Negotiation tactics":               ("NEGOTIATION TACTICS",                "BOTTOM LINE:"),
        "⭐ Bottom line":                       ("BOTTOM LINE:",                       None),
    }.items():
        if sm not in result:
            continue
        s = result.find(sm) + len(sm)
        e = result.find(em) if em and em in result else len(result)
        c = result[s:e].strip()
        if c:
            with st.expander(title, expanded=True):
                st.markdown(c)


# ─────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
def _clean_text_for_pdf(text: str) -> str:
    """Strip markdown syntax and emoji that ReportLab can't render."""
    import re
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    # Remove backticks
    text = re.sub(r'`+', '', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,4}\s*', '', text, flags=re.MULTILINE)
    # Strip emoji (basic unicode blocks)
    text = re.sub(
        r'[\U0001F300-\U0001FFFF\U00002600-\U000027FF\U0000FE00-\U0000FEFF]',
        '', text
    )
    # Collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _build_analysis_pdf(
    core_analysis: str,
    adv_results: dict,
    leverage_result: str,
    filename: str = "Legal_Guardian_Report",
    lang: str = "en",
) -> bytes:
    """Generate a compact but complete PDF summary of all analysis results."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=16,
        textColor=colors.HexColor("#0D2B55"),
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=colors.HexColor("#185FA5"),
        spaceBefore=10,
        spaceAfter=3,
        leading=14,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        spaceAfter=2,
    )
    verdict_style = ParagraphStyle(
        "Verdict",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#1A1A1A"),
        spaceBefore=4,
        spaceAfter=4,
        leading=13,
        borderPad=4,
    )

    story = []

    # ── Header ──────────────────────────────────────────
    title_text = "AI Legal Guardian" if lang == "en" else "AI कानूनी संरक्षक"
    sub_text   = "Document Analysis Report" if lang == "en" else "दस्तावेज़ विश्लेषण रिपोर्ट"
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph(sub_text, styles["Normal"]))
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#185FA5")))
    story.append(Spacer(1, 4 * mm))

    # ── Verdict banner ───────────────────────────────────
    verdict_map = {
        "Safe to Sign":          ("VERDICT: SAFE TO SIGN", colors.HexColor("#D4EDDA")),
        "Proceed with Caution":  ("VERDICT: PROCEED WITH CAUTION", colors.HexColor("#FFF3CD")),
        "Do NOT Sign":           ("VERDICT: DO NOT SIGN", colors.HexColor("#F8D7DA")),
        "हस्ताक्षर करना सुरक्षित": ("फैसला: हस्ताक्षर करना सुरक्षित", colors.HexColor("#D4EDDA")),
        "सावधानी से":            ("फैसला: सावधानी से आगे बढ़ें", colors.HexColor("#FFF3CD")),
        "हस्ताक्षर न करें":      ("फैसला: हस्ताक्षर न करें", colors.HexColor("#F8D7DA")),
    }
    verdict_label = "VERDICT: ANALYSIS COMPLETE"
    verdict_color = colors.HexColor("#E8F4FD")
    for keyword, (label, color_val) in verdict_map.items():
        if keyword in core_analysis:
            verdict_label = label
            verdict_color = color_val
            break

    verdict_table = [[Paragraph(f"<b>{verdict_label}</b>", verdict_style)]]
    vt = __import__('reportlab.platypus', fromlist=['Table', 'TableStyle'])
    from reportlab.platypus import Table as RLTable, TableStyle as RLTableStyle
    t = RLTable(verdict_table, colWidths=["100%"])
    t.setStyle(RLTableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_color),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#AAAAAA")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # ── Helper: add a section ────────────────────────────
    def _add_section(heading: str, raw_text: str, char_limit: int = 1200):
        story.append(Paragraph(heading, section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
        clean = _clean_text_for_pdf(raw_text)
        # Truncate very long sections to keep the PDF compact
        if len(clean) > char_limit:
            clean = clean[:char_limit].rsplit("\n", 1)[0] + "\n..."
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 1.5 * mm))
                continue
            try:
                story.append(Paragraph(line, body_style))
            except Exception:
                story.append(Paragraph(line.encode("ascii", "ignore").decode(), body_style))

    # ── Core analysis sections ───────────────────────────
    section_map_en = {
        "Summary":         ("SUMMARY:",         "RISKS:"),
        "Key Risks":       ("RISKS:",           "OBLIGATIONS:"),
        "Obligations":     ("OBLIGATIONS:",     "FRAUD CHECK:"),
        "Fraud Check":     ("FRAUD CHECK:",     "TAMPERING CHECK:"),
        "Missing Clauses": ("MISSING CLAUSES:", "FINAL DECISION:"),
        "Final Decision":  ("FINAL DECISION:",  None),
    }
    section_map_hi = {
        "सारांश":            ("📄 सारांश:",         "⚠️ जोखिम:"),
        "मुख्य जोखिम":       ("⚠️ जोखिम:",          "📌 दायित्व:"),
        "दायित्व":           ("📌 दायित्व:",         "🚨 धोखाधड़ी जांच:"),
        "धोखाधड़ी जांच":      ("🚨 धोखाधड़ी जांच:",   "🔍 छेड़छाड़ जांच:"),
        "अनुपस्थित धाराएं":  ("📉 अनुपस्थित धाराएं:", "🤖 अंतिम निर्णय:"),
        "अंतिम निर्णय":      ("🤖 अंतिम निर्णय:",    None),
    }
    section_map = section_map_en if lang == "en" else section_map_hi

    for heading, (sm, em) in section_map.items():
        if sm not in core_analysis:
            continue
        s = core_analysis.find(sm) + len(sm)
        e = core_analysis.find(em) if em and em in core_analysis else len(core_analysis)
        content = core_analysis[s:e].strip()
        if content:
            _add_section(heading, content, char_limit=900)

    story.append(Spacer(1, 3 * mm))

    # ── Advanced modules (brief) ─────────────────────────
    adv_headings = {
        "heatmap":     ("Heatmap Bias",          "हीटमैप पूर्वाग्रह"),
        "future_risk": ("Future Risk Scenarios",  "भविष्य जोखिम"),
        "negotiation": ("Smart Negotiation",      "स्मार्ट वार्ता"),
        "kfs":         ("KFS Integrity Audit",    "KFS ऑडिट"),
        "prepayment":  ("Pre-Payment Shield",     "पूर्व-भुगतान"),
        "lc":          ("LC Discrepancy",         "LC विसंगति"),
        "colending":   ("Co-Lending Radar",       "को-लेंडिंग रडार"),
    }
    for key, (en_h, hi_h) in adv_headings.items():
        if key not in adv_results:
            continue
        heading = en_h if lang == "en" else hi_h
        _add_section(heading, adv_results[key], char_limit=600)

    # ── Leverage ─────────────────────────────────────────
    if leverage_result:
        lev_heading = "Leverage Radar" if lang == "en" else "लीवरेज रडार"
        # Extract just the scorecard summary (first ~600 chars)
        _add_section(lev_heading, leverage_result, char_limit=600)

    # ── Footer ───────────────────────────────────────────
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
    footer_txt = (
        "Generated by AI Legal Guardian  |  Powered by Groq gpt-oss-20b  |  "
        "For informational purposes only — not legal advice."
    ) if lang == "en" else (
        "AI कानूनी संरक्षक द्वारा निर्मित  |  Groq gpt-oss-20b  |  "
        "केवल जानकारी के लिए — कानूनी सलाह नहीं।"
    )
    story.append(Paragraph(footer_txt, ParagraphStyle(
        "Footer", parent=styles["Normal"], fontSize=6.5,
        textColor=colors.HexColor("#888888"), leading=9,
    )))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
# AUTONOMOUS PIPELINE
# ─────────────────────────────────────────────────────────────
PIPELINE = [
    ("heatmap",     "🔥 Heatmap Bias Flag",         run_heatmap_bias),
    ("future_risk", "🔮 Future Risk Simulator",     run_future_risk),
    ("timeline",    "📈 Risk Timeline",              run_risk_timeline),
    ("negotiation", "🤝 Smart Negotiation",         run_smart_negotiation),
    ("kfs",         "🧾 KFS Integrity Audit",       run_kfs_audit),
    ("prepayment",  "🏦 Pre-Payment & Foreclosure", run_prepayment),
    ("lc",          "📑 LC Discrepancy Detector",   run_lc_discrepancy),
    ("colending",   "⚖️ Co-Lending Radar",          run_co_lending),
]
PAUSE_BETWEEN = 8
PAUSE_AFTER_CORE = 10

if st.session_state.doc_text and not st.session_state.auto_done:
    prog   = st.progress(0)
    status = st.empty()
    total  = 1 + len(PIPELINE) + 1

    status.info(f"{L['step_core']} (1/10)")
    try:
        st.session_state.core_analysis = run_full_analysis(st.session_state.doc_text, lang)
    except Exception as e:
        st.session_state.core_analysis = f"⚠️ Error: {e}"
    prog.progress(1 / total)
    time.sleep(PAUSE_AFTER_CORE)

    for i, (key, label, fn) in enumerate(PIPELINE):
        status.info(f"🤖 {label}... ({i + 2}/10)")
        try:
            st.session_state.adv_results[key] = fn(st.session_state.doc_text, lang)
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                status.warning(L["rate_wait"].format(label))
                time.sleep(30)
                try:
                    st.session_state.adv_results[key] = fn(st.session_state.doc_text, lang)
                except Exception as e2:
                    st.session_state.adv_results[key] = f"⚠️ Module error: {e2}"
            else:
                st.session_state.adv_results[key] = f"⚠️ Module error: {e}"
        prog.progress((i + 2) / total)
        if i < len(PIPELINE) - 1:
            time.sleep(PAUSE_BETWEEN)

    status.info(L["step_leverage"])
    try:
        st.session_state.leverage_result = run_leverage_mapping(st.session_state.doc_text, lang)
    except Exception as e:
        err = str(e)
        if "rate_limit" in err.lower() or "429" in err:
            status.warning(L["rate_wait"].format("Leverage Radar"))
            time.sleep(30)
            try:
                st.session_state.leverage_result = run_leverage_mapping(st.session_state.doc_text, lang)
            except Exception as e2:
                st.session_state.leverage_result = f"⚠️ Leverage error: {e2}"
        else:
            st.session_state.leverage_result = f"⚠️ Leverage error: {e}"

    prog.progress(1.0)
    st.session_state.auto_done = True
    status.success(L["step_done"])


# ─────────────────────────────────────────────────────────────
# TAB 1 — FULL ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab1:
    if not st.session_state.doc_text:
        st.info(L["upload_prompt"])
    elif not st.session_state.auto_done:
        st.info(L["running_prompt"])
    else:
        adv = st.session_state.adv_results
        _render_core(st.session_state.core_analysis)
        st.divider()

        if "heatmap" in adv:
            with st.expander(L["adv_expanders"]["heatmap"], expanded=True):
                st.markdown(adv["heatmap"])
            st.divider()

        if "future_risk" in adv or "timeline" in adv:
            with st.expander(L["adv_expanders"]["future_risk"], expanded=True):
                if "future_risk" in adv:
                    st.markdown(adv["future_risk"])
                if "timeline" in adv:
                    st.markdown("---")
                    st.markdown(adv["timeline"])
            st.divider()

        if "negotiation" in adv:
            with st.expander(L["adv_expanders"]["negotiation"], expanded=True):
                st.markdown(adv["negotiation"])
            st.divider()

        if "kfs" in adv:
            with st.expander(L["adv_expanders"]["kfs"], expanded=True):
                st.markdown(adv["kfs"])
            st.divider()

        if "prepayment" in adv:
            with st.expander(L["adv_expanders"]["prepayment"], expanded=True):
                st.markdown(adv["prepayment"])
            st.divider()

        if "lc" in adv:
            with st.expander(L["adv_expanders"]["lc"], expanded=True):
                st.markdown(adv["lc"])
            st.divider()

        if "colending" in adv:
            with st.expander(L["adv_expanders"]["colending"], expanded=True):
                st.markdown(adv["colending"])
            st.divider()

        clauses = _extract_risky_clauses(st.session_state.core_analysis)
        if clauses:
            with st.expander(L["ghost_expander"], expanded=False):
                st.caption(L["ghost_caption"])
                for i, clause in enumerate(clauses):
                    c1, c2 = st.columns([4, 1])
                    c1.markdown(f"`{clause['label']}`")
                    if c2.button(L["ghost_fix"], key=f"ghost_{i}"):
                        with st.spinner("..."):
                            st.session_state.ghost_results[clause["label"]] = run_ghost_negotiator(
                                st.session_state.doc_text, clause["full"], lang)
                for label, gr in st.session_state.ghost_results.items():
                    st.markdown(f"**{label}**")
                    _render_ghost(gr)
                    st.divider()

        # ── PDF Download ───────────────────────────────
        st.divider()
        col_pdf1, col_pdf2 = st.columns([3, 1])
        with col_pdf2:
            if st.button(L["pdf_download_btn"], use_container_width=True, type="primary"):
                with st.spinner(L["pdf_generating"]):
                    try:
                        pdf_bytes = _build_analysis_pdf(
                            core_analysis=st.session_state.core_analysis,
                            adv_results=st.session_state.adv_results,
                            leverage_result=st.session_state.leverage_result,
                            lang=lang,
                        )
                        st.session_state.analysis_pdf_bytes = pdf_bytes
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")

        if st.session_state.analysis_pdf_bytes:
            st.download_button(
                label=L["pdf_ready"],
                data=st.session_state.analysis_pdf_bytes,
                file_name="Legal_Guardian_Analysis.pdf",
                mime="application/pdf",
                use_container_width=False,
            )


# ─────────────────────────────────────────────────────────────
# TAB 2 — LEVERAGE RADAR
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader(L["leverage_sub"])
    st.caption(L["leverage_caption"])
    if not st.session_state.doc_text:
        st.info(L["leverage_upload"])
    elif not st.session_state.auto_done:
        st.info(L["leverage_running"])
    else:
        if st.session_state.leverage_result:
            _render_leverage(st.session_state.leverage_result)
            with st.expander(L["raw_output"]):
                st.text(st.session_state.leverage_result)
        else:
            st.warning(L["leverage_no_result"])


# ─────────────────────────────────────────────────────────────
# TAB 3 — CHATBOT
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader(L["chat_sub"])
    st.caption(L["chat_caption"])
    if not st.session_state.doc_text:
        st.info(L["chat_upload"])
    else:
        suggestions = get_scenario_suggestions(lang)
        if not st.session_state.chat_history:
            st.markdown(L["chat_tap"])
            cols = st.columns(4)
            for i, sug in enumerate(suggestions):
                if cols[i % 4].button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.scenario_input = sug
                    st.rerun()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.scenario_input:
            injected = st.session_state.scenario_input
            st.session_state.scenario_input = ""
            st.session_state.chat_history.append({"role": "user", "content": injected})
            with st.chat_message("user"):
                st.markdown(injected)
            with st.chat_message("assistant"):
                with st.spinner("..."):
                    # Detect "what if" in both languages
                    is_scenario = injected.lower().startswith("what if") or injected.startswith("अगर")
                    ans = run_scenario(st.session_state.doc_text, injected, lang) \
                        if is_scenario else run_qa(st.session_state.doc_text, injected, lang)
                st.markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})

        user_input = st.chat_input(L["chat_input"], key="chat_tab_input")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("..."):
                    is_scenario = user_input.lower().startswith("what if") or user_input.startswith("अगर")
                    ans = run_scenario(st.session_state.doc_text, user_input, lang) \
                        if is_scenario else run_qa(st.session_state.doc_text, user_input, lang)
                st.markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})

        if st.session_state.chat_history:
            st.divider()
            if st.button(L["clear_chat"]):
                st.session_state.chat_history = []
                st.rerun()


# ─────────────────────────────────────────────────────────────
# TAB 4 — AI COURTROOM
# ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader(L["court_sub"])
    st.caption(L["court_caption"])
    st.divider()

    st.markdown(L["court_select_case"])
    card_cols = st.columns(3)

    for col, key in zip(card_cols, CASE_KEYS):
        case = COURTROOM_CASES[key]
        is_active = st.session_state.court_case_key == key
        border_color = "#185FA5" if is_active else "#e0e0e0"
        bg_color     = "#EBF4FD" if is_active else "#FAFAFA"

        title_disp    = get_case_field(key, "title",    lang)
        category_disp = get_case_field(key, "category", lang)
        summary_disp  = get_case_field(key, "summary",  lang)
        charge_disp   = get_case_field(key, "charge",   lang)

        with col:
            st.markdown(
                f"""
                <div style="
                    border: {'2px' if is_active else '1px'} solid {border_color};
                    border-radius: 12px;
                    padding: 1rem 1.1rem;
                    background: {bg_color};
                    min-height: 160px;
                ">
                    <span style="font-size:12px; font-weight:600; color:#555;">{category_disp}</span>
                    <p style="font-weight:600; font-size:15px; margin: 6px 0 4px;">{title_disp}</p>
                    <p style="font-size:12px; color:#666; line-height:1.45;">{summary_disp}</p>
                    <p style="font-size:11px; color:#999; margin-top:8px; font-style:italic;">{charge_disp}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            btn_label = L["court_selected"] if is_active else L["court_select_btn"]
            if st.button(btn_label, key=f"court_select_{key}", use_container_width=True, disabled=is_active):
                st.session_state.court_case_key = key
                intro = get_case_field(key, "intro", lang)
                st.session_state.court_history = [{"role": "assistant", "content": intro}]
                st.session_state.court_suggestion_clicked = ""
                st.rerun()

    st.divider()

    if not st.session_state.court_case_key:
        st.info(L["court_pick_prompt"])
    else:
        active_key = st.session_state.court_case_key
        title_disp  = get_case_field(active_key, "title",  lang)
        charge_disp = get_case_field(active_key, "charge", lang)

        atty_col1, atty_col2 = st.columns([1, 11])
        with atty_col1:
            st.markdown(
                "<div style='width:44px;height:44px;border-radius:50%;background:#E6F1FB;"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:22px;'>⚖️</div>",
                unsafe_allow_html=True,
            )
        with atty_col2:
            st.markdown(
                f"**AI Defence Attorney** &nbsp;|&nbsp; "
                f"<span style='color:#185FA5;font-size:13px;'>{title_disp}</span><br>"
                f"<span style='font-size:12px;color:#888;'>{charge_disp}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        for msg in st.session_state.court_history:
            with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
                st.markdown(msg["content"])

        if len(st.session_state.court_history) == 1:
            st.markdown(L["court_quick"])
            suggestions_court = get_case_field(active_key, "suggestions", lang)
            if isinstance(suggestions_court, list):
                sug_cols = st.columns(3)
                for i, sug in enumerate(suggestions_court):
                    if sug_cols[i % 3].button(sug, key=f"court_sug_{i}", use_container_width=True):
                        st.session_state.court_suggestion_clicked = sug
                        st.rerun()

        if st.session_state.court_suggestion_clicked:
            injected = st.session_state.court_suggestion_clicked
            st.session_state.court_suggestion_clicked = ""
            st.session_state.court_history.append({"role": "user", "content": injected})
            with st.chat_message("user"):
                st.markdown(injected)
            with st.chat_message("assistant"):
                with st.spinner(L["court_thinking"]):
                    try:
                        reply = run_courtroom_chat(active_key, st.session_state.court_history, lang)
                    except Exception as e:
                        reply = f"⚠️ {e}"
                st.markdown(reply)
            st.session_state.court_history.append({"role": "assistant", "content": reply})
            st.rerun()

        user_msg = st.chat_input(L["chat_input"], key="court_tab_input")
        if user_msg:
            st.session_state.court_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                with st.spinner(L["court_thinking"]):
                    try:
                        reply = run_courtroom_chat(active_key, st.session_state.court_history, lang)
                    except Exception as e:
                        err = str(e)
                        if "rate_limit" in err.lower() or "429" in err:
                            time.sleep(20)
                            try:
                                reply = run_courtroom_chat(active_key, st.session_state.court_history, lang)
                            except Exception as e2:
                                reply = f"⚠️ {e2}"
                        else:
                            reply = f"⚠️ {e}"
                st.markdown(reply)
            st.session_state.court_history.append({"role": "assistant", "content": reply})

        if len(st.session_state.court_history) > 1:
            st.divider()
            cl1, cl2 = st.columns([6, 1])
            with cl2:
                if st.button(L["court_reset"], use_container_width=True):
                    intro = get_case_field(active_key, "intro", lang)
                    st.session_state.court_history = [{"role": "assistant", "content": intro}]
                    st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB 5 — FORM FILLER AGENT
# ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader(L["formfill_sub"])
    st.caption(L["formfill_caption"])
    st.divider()

    # Separate uploader just for this tab
    ff_uploaded = st.file_uploader(
        L["formfill_uploader"],
        type=["pdf", "docx", "txt"],
        key="formfill_uploader_widget",
    )

    # When a new form is uploaded → extract text + auto-run agent
    if ff_uploaded and ff_uploaded.name != st.session_state.formfill_last_file:
        ext = ff_uploaded.name.split(".")[-1].lower()
        with st.spinner("..."):
            st.session_state.formfill_text = extract_text(ff_uploaded, ext)
        st.session_state.formfill_last_file = ff_uploaded.name
        st.session_state.formfill_result = ""   # reset so we re-run below

    # Show loaded info
    if st.session_state.formfill_text:
        wc = len(st.session_state.formfill_text.split())
        st.success(
            f"{L['formfill_loaded']}: **{st.session_state.formfill_last_file}** "
            f"— {L['formfill_words']}: {wc:,}"
        )

        # Auto-run analysis if result not yet generated
        if not st.session_state.formfill_result:
            with st.spinner(L["formfill_running"]):
                try:
                    st.session_state.formfill_result = run_formfill_analysis(
                        st.session_state.formfill_text, lang
                    )
                except Exception as e:
                    err = str(e)
                    if "rate_limit" in err.lower() or "429" in err:
                        time.sleep(20)
                        try:
                            st.session_state.formfill_result = run_formfill_analysis(
                                st.session_state.formfill_text, lang
                            )
                        except Exception as e2:
                            st.session_state.formfill_result = f"⚠️ {e2}"
                    else:
                        st.session_state.formfill_result = f"⚠️ {e}"

        # Display results
        if st.session_state.formfill_result:
            result_text = st.session_state.formfill_result

            if result_text.startswith("⚠️"):
                st.error(result_text)
            elif (
                "No form fields detected" in result_text
                or "No form fields" in result_text
                or "कोई फ़ॉर्म फ़ील्ड नहीं" in result_text
            ):
                st.warning(L["formfill_no_result"])
                st.info(result_text)
            else:
                # Render the guide nicely
                # Split on numbered fields for individual expanders
                import re as _re
                field_blocks = _re.split(r'\n(?=\d+\.\s)', result_text)

                # First block may contain the header line before field 1
                header_block = ""
                if field_blocks and not _re.match(r'^\d+\.', field_blocks[0].strip()):
                    header_block = field_blocks.pop(0)

                if header_block.strip():
                    st.markdown(header_block.strip())

                st.markdown("---")

                for block in field_blocks:
                    block = block.strip()
                    if not block:
                        continue
                    # Extract title from first line (e.g. "1. Field Name:")
                    lines = block.split("\n")
                    title = lines[0].strip()
                    body  = "\n".join(lines[1:]).strip()

                    with st.expander(title, expanded=True):
                        st.markdown(body)

                # Quick summary and notes — look for them after last field
                for marker in ["📋 Quick Summary:", "📋 त्वरित सारांश:", "⚠️ Important Notes:", "⚠️ महत्वपूर्ण नोट:"]:
                    if marker in result_text:
                        s = result_text.find(marker)
                        snippet = result_text[s:].strip()
                        st.divider()
                        st.info(snippet[:800])
                        break

        # Reset button
        st.divider()
        if st.button(L["formfill_rerun"], key="formfill_reset"):
            st.session_state.formfill_text = ""
            st.session_state.formfill_result = ""
            st.session_state.formfill_last_file = ""
            st.rerun()
    else:
        st.info(L["formfill_upload_prompt"])


# ─────────────────────────────────────────────────────────────
# TAB 6 — LANGGRAPH + RAG  (NEW)
# ─────────────────────────────────────────────────────────────
with tab6:
    st.subheader(L["lg_sub"])
    st.caption(L["lg_caption"])
    st.divider()

    # ── Topology diagram ─────────────────────────────────────────────
    with st.expander("🗺️ Pipeline topology", expanded=False):
        st.markdown(
            "```\n"
            "  START\n"
            "    │\n"
            "    ▼\n"
            "  ingest_doc  ───►  chunks → Chroma vector store + SQLite mirror\n"
            "    │\n"
            "    ▼\n"
            "  core_analysis  (risk / fraud / obligations / verdict)\n"
            "    │\n"
            "    ▼\n"
            "  leverage  ───►  parse leverage_score (0–100)\n"
            "    │\n"
            "    ▼  (conditional edge)\n"
            "    ├── score ≥ 60 ─► future_risk\n"
            "    ├── 40 ≤ score < 60 ─► smart_negotiation\n"
            "    └── score < 40  ─► ghost_negotiator\n"
            "    │\n"
            "    ▼\n"
            "  heatmap_bias  (clause bias detection)\n"
            "    │\n"
            "    ▼\n"
            "  rag_qa  ───►  retrieve top-4 chunks → LLM grounded answer\n"
            "    │\n"
            "    ▼\n"
            "  finalize  ───►  persist full state to graph_runs table\n"
            "    │\n"
            "    ▼\n"
            "   END\n"
            "```"
        )

    # ── Run button ───────────────────────────────────────────────────
    col_run1, col_run2 = st.columns([3, 2])
    with col_run1:
        run_btn = st.button(L["lg_run_btn"], type="primary",
                             disabled=not st.session_state.doc_text,
                             use_container_width=True)
    with col_run2:
        if st.session_state.db_doc_id:
            st.caption(f"DB doc id: `{st.session_state.db_doc_id[:8]}…`")
        if st.session_state.lg_final_state:
            st.caption(f"Last run status: **{st.session_state.lg_final_state.get('status', '—')}**")

    if not st.session_state.doc_text:
        st.info(L["lg_upload_prompt"])
    elif run_btn:
        # ── Persist the doc to DB if not already done (e.g. user skipped sidebar upload)
        if not st.session_state.db_doc_id:
            try:
                dh = hashlib.sha256(st.session_state.doc_text.encode("utf-8")).hexdigest()
                doc_row = database.upsert_document(
                    filename=st.session_state.last_file or "pasted.txt",
                    file_type="txt",
                    raw_text=st.session_state.doc_text,
                    language=lang,
                    doc_hash=dh,
                )
                st.session_state.db_doc_id = doc_row.id
                st.session_state.db_doc_hash = dh
            except Exception as e:
                st.error(f"Failed to persist document: {e}")

        st.session_state.lg_progress = []
        st.session_state.lg_running = True
        progress_bar = st.progress(0.0, text=L["lg_running"])
        trace_box = st.container()

        def _on_step(node_name: str, partial: dict):
            # Append to running trace (deduped by node name since LangGraph
            # typically emits once per node).
            if partial.get("trace"):
                st.session_state.lg_progress = partial["trace"]
            with trace_box:
                st.info(f"✓ Node finished: **{node_name}**")

        try:
            final_state = langgraph_orchestrator.run_pipeline(
                document_id=st.session_state.db_doc_id,
                document_text=st.session_state.doc_text,
                language=lang,
                on_step=_on_step,
            )
            st.session_state.lg_final_state = final_state
            progress_bar.progress(1.0, text=L["lg_done"])
            st.success(L["lg_done"])
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.session_state.lg_final_state = {"status": "error", "error": str(e)}
        finally:
            st.session_state.lg_running = False

    # ── Render final state ───────────────────────────────────────────
    fs = st.session_state.lg_final_state
    if fs:
        st.divider()
        st.markdown(f"**Status:** `{fs.get('status', '—')}`")

        # ── Trace table ─────────────────────────────────────────────
        if fs.get("trace"):
            st.markdown(f"#### {L['lg_trace_label']}")
            trace_data = [
                {
                    "Node": t.get("node", ""),
                    "OK": "✅" if t.get("ok") else "❌",
                    "Duration (ms)": t.get("duration_ms", 0),
                    "Error": (t.get("error") or "")[:60],
                }
                for t in fs["trace"]
            ]
            st.dataframe(trace_data, use_container_width=True, hide_index=True)

        # ── Branch taken ─────────────────────────────────────────────
        if fs.get("branch_name"):
            st.info(f"{L['lg_branch_label']}: **{fs['branch_name']}**  "
                    f"(leverage score = {fs.get('leverage_score', '—')})")

        # ── Node outputs ─────────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            if fs.get("core_analysis"):
                with st.expander("📄 Core analysis (from LangGraph node)", expanded=False):
                    st.markdown(fs["core_analysis"])
            if fs.get("leverage_analysis"):
                with st.expander("📡 Leverage analysis", expanded=False):
                    st.markdown(fs["leverage_analysis"])
        with col_b:
            if fs.get("branch_output"):
                title_map = {
                    "future_risk":      "🔮 Future Risk Simulator (branch)",
                    "negotiation":      "🤝 Smart Negotiation (branch)",
                    "ghost_negotiator": "👻 Ghost Negotiator (branch)",
                }
                with st.expander(title_map.get(fs.get("branch_name", ""),
                                                "Branch output"), expanded=False):
                    st.markdown(fs["branch_output"])
            if fs.get("heatmap_bias"):
                with st.expander("🔥 Heatmap Bias Flag", expanded=False):
                    st.markdown(fs["heatmap_bias"])

        # ── RAG answer ───────────────────────────────────────────────
        ra = fs.get("rag_answer") or {}
        if ra:
            st.markdown(f"#### {L['lg_rag_q_label']}")
            st.info(f"_{langgraph_orchestrator.DEFAULT_RAG_QUESTION}_")
            st.markdown(ra.get("answer", "_(no answer)_"))
            src_ids = [s for s in (ra.get("sources") or [])]
            if src_ids:
                st.caption(f"Sources: {', '.join(map(str, src_ids))}")

    # ── RAG chat (interactive, persistent) ────────────────────────────
    st.divider()
    st.markdown(f"#### {L['lg_chat_sub']}")
    st.caption(L["lg_chat_caption"])

    if not st.session_state.doc_text:
        st.info(L["lg_upload_prompt"])
    elif not st.session_state.db_doc_id:
        st.warning("Document not yet persisted — click **Run Pipeline** once to ingest chunks.")
    else:
        # Create or reuse a chat session for this doc + tab
        if not st.session_state.rag_session_id:
            try:
                sess = database.create_chat_session(
                    document_id=st.session_state.db_doc_id,
                    mode="qa_rag",
                    title=f"RAG chat · {st.session_state.last_file or 'pasted'}",
                    language=lang,
                )
                st.session_state.rag_session_id = sess.id
            except Exception as e:
                st.error(f"Failed to create chat session: {e}")

        # Render history
        for msg in st.session_state.rag_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(L["lg_chat_sources"], expanded=False):
                        for s in msg["sources"]:
                            st.markdown(f"- `{s['chunk_id']}` (score={s['score']:.3f}) — "
                                        f"_{s['text'][:140].strip()}…_")

        # Chat input
        q = st.chat_input(L["lg_chat_input"], key="rag_chat_input")
        if q:
            st.session_state.rag_chat_history.append(
                {"role": "user", "content": q, "sources": []}
            )
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving chunks + generating grounded answer…"):
                    try:
                        res = rag_engine.answer_with_rag(
                            document_id=st.session_state.db_doc_id,
                            question=q, k=4, temperature=0.1,
                        )
                        answer = res["answer"]
                        sources = res["sources"]
                        # Persist to DB
                        if st.session_state.rag_session_id:
                            try:
                                database.append_message(
                                    session_id=st.session_state.rag_session_id,
                                    role="user", content=q,
                                )
                                database.append_message(
                                    session_id=st.session_state.rag_session_id,
                                    role="assistant", content=answer,
                                    retrieved_chunks=[
                                        {"chunk_id": s["chunk_id"],
                                         "score": s["score"]} for s in sources
                                    ],
                                )
                            except Exception as e:
                                print(f"[DB] chat persistence failed: {e}")
                    except Exception as e:
                        answer = f"⚠️ RAG failed: {e}"
                        sources = []
                st.markdown(answer)
                if sources:
                    with st.expander(L["lg_chat_sources"], expanded=False):
                        for s in sources:
                            st.markdown(f"- `{s['chunk_id']}` (score={s['score']:.3f}) — "
                                        f"_{s['text'][:140].strip()}…_")
            st.session_state.rag_chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )


# ─────────────────────────────────────────────────────────────
# TAB 7 — DATABASE / HISTORY  (NEW)
# ─────────────────────────────────────────────────────────────
with tab7:
    st.subheader(L["db_sub"])
    st.caption(L["db_caption"])
    st.divider()

    # ── Documents ─────────────────────────────────────────────────────
    st.markdown(f"#### {L['db_documents_h']}")
    try:
        docs = database.list_documents(limit=50)
    except Exception as e:
        docs = []
        st.error(f"DB read failed: {e}")

    if not docs:
        st.info(L["db_no_docs"])
    else:
        rows = []
        for d in docs:
            rows.append({
                "id": d.id[:8] + "…",
                "filename": d.filename,
                "type": d.file_type,
                "words": d.word_count,
                "lang": d.language,
                "uploaded": d.created_at.strftime("%Y-%m-%d %H:%M") if d.created_at else "",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.write("")
        sel_id_short = st.selectbox(
            "Select document (by short id) to inspect / reload",
            options=[r["id"] for r in rows],
            key="db_doc_select",
        )
        col_db1, col_db2, col_db3 = st.columns(3)
        with col_db1:
            if st.button("🔁 Reload into app", key="db_reload_btn"):
                # Find full id
                full_id = next(
                    (d.id for d in docs if d.id.startswith(sel_id_short.rstrip("…"))),
                    None,
                )
                if full_id:
                    d = database.get_document(full_id)
                    if d:
                        st.session_state.doc_text = d.raw_text
                        st.session_state.last_file = d.filename
                        st.session_state.db_doc_id = d.id
                        st.session_state.db_doc_hash = d.doc_hash
                        st.session_state.auto_done = False
                        st.session_state.core_analysis = ""
                        st.session_state.adv_results = {}
                        st.session_state.leverage_result = ""
                        st.session_state.chat_history = []
                        st.session_state.rag_chat_history = []
                        st.session_state.rag_session_id = None
                        st.success(f"Reloaded `{d.filename}` into the app.")
        with col_db2:
            if st.button("🔍 View analyses", key="db_view_an_btn"):
                st.session_state["db_show_doc_id"] = (
                    next((d.id for d in docs if d.id.startswith(sel_id_short.rstrip("…"))), None)
                )
        with col_db3:
            if st.button("🧩 View RAG chunks", key="db_view_chunks_btn"):
                st.session_state["db_show_chunks_doc_id"] = (
                    next((d.id for d in docs if d.id.startswith(sel_id_short.rstrip("…"))), None)
                )

    # ── Analyses for selected doc ─────────────────────────────────────
    show_doc_id = st.session_state.get("db_show_doc_id")
    if show_doc_id:
        st.divider()
        st.markdown(f"#### {L['db_analyses_h']} · `{show_doc_id[:8]}…`")
        an_rows = database.list_analyses(show_doc_id)
        if not an_rows:
            st.info("No analyses stored for this document yet.")
        else:
            data = [{
                "module": a.module,
                "language": a.language,
                "leverage_score": a.leverage_score,
                "duration_ms": a.duration_ms,
                "created_at": a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "",
                "result_preview": (a.result_text or "")[:200],
            } for a in an_rows]
            st.dataframe(data, use_container_width=True, hide_index=True)
            # Show full text of latest analysis
            latest = an_rows[0]
            with st.expander(f"Full text — {latest.module} ({latest.created_at})", expanded=False):
                st.markdown(latest.result_text)

    # ── RAG chunks for selected doc ───────────────────────────────────
    chunks_doc_id = st.session_state.get("db_show_chunks_doc_id")
    if chunks_doc_id:
        st.divider()
        st.markdown(f"#### 🧩 RAG chunks · `{chunks_doc_id[:8]}…`")
        try:
            with database.session_scope() as s:
                from sqlalchemy import select
                stmt = select(database.RagChunk).where(
                    database.RagChunk.document_id == chunks_doc_id
                ).order_by(database.RagChunk.chunk_index)
                chunks = s.execute(stmt).scalars().all()
            if not chunks:
                st.info("No chunks persisted yet. Run the LangGraph pipeline to ingest.")
            else:
                chunk_rows = [{
                    "index": c.chunk_index,
                    "chroma_id": c.chroma_id,
                    "tokens": c.token_count,
                    "preview": c.chunk_text[:120],
                } for c in chunks]
                st.dataframe(chunk_rows, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Failed to load chunks: {e}")

    # ── Graph runs ────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"#### {L['db_runs_h']}")
    try:
        runs = database.list_graph_runs(limit=20)
    except Exception:
        runs = []
    if not runs:
        st.info("No LangGraph runs persisted yet.")
    else:
        run_rows = [{
            "id": r.id[:8] + "…",
            "label": r.run_label,
            "status": r.status,
            "doc_id": (r.document_id or "")[:8] + "…" if r.document_id else "",
            "started_at": r.started_at.strftime("%Y-%m-%d %H:%M:%S") if r.started_at else "",
            "ended_at": r.ended_at.strftime("%H:%M:%S") if r.ended_at else "",
        } for r in runs]
        st.dataframe(run_rows, use_container_width=True, hide_index=True)

        sel_run_short = st.selectbox(
            "Inspect run trace",
            options=[r["id"] for r in run_rows],
            key="db_run_select",
        )
        if sel_run_short:
            full_run_id = next(
                (r.id for r in runs if r.id.startswith(sel_run_short.rstrip("…"))), None
            )
            if full_run_id:
                run = next((r for r in runs if r.id == full_run_id), None)
                if run:
                    import json as _json
                    try:
                        trace = _json.loads(run.trace_json or "[]")
                        if trace:
                            tr_df = [{
                                "node": t.get("node"),
                                "ok": t.get("ok"),
                                "duration_ms": t.get("duration_ms"),
                            } for t in trace]
                            st.dataframe(tr_df, use_container_width=True, hide_index=True)
                    except Exception:
                        pass
                    try:
                        state = _json.loads(run.state_json or "{}")
                        with st.expander("Final state JSON", expanded=False):
                            st.json(state)
                    except Exception:
                        pass

    # ── Chat sessions ─────────────────────────────────────────────────
    st.divider()
    st.markdown(f"#### {L['db_chats_h']}")
    try:
        sessions = database.list_chat_sessions(limit=20)
    except Exception:
        sessions = []
    if not sessions:
        st.info("No chat sessions persisted yet.")
    else:
        sess_rows = [{
            "id": s.id[:8] + "…",
            "mode": s.mode,
            "title": s.title,
            "lang": s.language,
            "updated_at": s.updated_at.strftime("%Y-%m-%d %H:%M") if s.updated_at else "",
        } for s in sessions]
        st.dataframe(sess_rows, use_container_width=True, hide_index=True)

        sel_sess_short = st.selectbox(
            "Inspect chat session",
            options=[r["id"] for r in sess_rows],
            key="db_sess_select",
        )
        if sel_sess_short:
            full_sess_id = next(
                (s.id for s in sessions if s.id.startswith(sel_sess_short.rstrip("…"))), None
            )
            if full_sess_id:
                msgs = database.list_messages(full_sess_id)
                if msgs:
                    for m in msgs:
                        with st.chat_message("assistant" if m.role == "assistant" else "user"):
                            st.markdown(m.content)
                            if m.retrieved_chunks:
                                with st.expander("Retrieved chunks", expanded=False):
                                    st.code(m.retrieved_chunks)
                else:
                    st.info("This session has no messages yet.")