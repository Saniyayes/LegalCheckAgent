"""
agent_courtroom.py — AI Courtroom Agent
Responsible for: Defence attorney chatbot for 3 sample legal cases.
Always argues in the client's favour. Never admits guilt.
Called by the orchestrator in agent.py.
Supports lang="en" and lang="hi".

LLM backend: uses llm_adapter (Groq if configured, else Z-AI GLM-4-plus).
"""
from llm_adapter import chat_with_history as _call_llm_history


# ── Case definitions (bilingual) ──────────────────────────────────────────────
COURTROOM_CASES = {
    "case_1": {
        "key": "case_1",
        "title_en": "State v. Arjun Mehta",
        "title_hi": "राज्य बनाम अर्जुन मेहता",
        "category_en": "🔴 Criminal",
        "category_hi": "🔴 आपराधिक",
        "charge_en": "Section 379 IPC — Theft",
        "charge_hi": "धारा 379 IPC — चोरी",
        "summary_en": (
            "Arjun Mehta is accused of theft at a retail store. CCTV footage is blurry "
            "and inconclusive. He has no prior criminal record. The arresting officer "
            "failed to follow proper CrPC procedure during the arrest."
        ),
        "summary_hi": (
            "अर्जुन मेहता पर एक खुदरा दुकान में चोरी का आरोप है। CCTV फुटेज धुंधली और "
            "अनिर्णायक है। उनका कोई पूर्व आपराधिक रिकॉर्ड नहीं है। गिरफ्तारी के दौरान "
            "अधिकारी ने CrPC की उचित प्रक्रिया का पालन नहीं किया।"
        ),
        "intro_en": (
            "I'm your defense attorney for the Section 379 IPC theft charge against you. "
            "The prosecution's case is weak — blurry CCTV is not conclusive evidence, "
            "and a procedurally flawed arrest can be challenged in court. "
            "Everything you tell me is protected under attorney-client privilege. "
            "Walk me through exactly what happened that day."
        ),
        "intro_hi": (
            "मैं धारा 379 IPC चोरी के आरोप में आपका बचाव वकील हूं। "
            "अभियोजन पक्ष का मामला कमजोर है — धुंधली CCTV निर्णायक सबूत नहीं है, "
            "और प्रक्रियागत रूप से दोषपूर्ण गिरफ्तारी को अदालत में चुनौती दी जा सकती है। "
            "आप मुझे जो भी बताएंगे वह वकील-मुवक्किल विशेषाधिकार के तहत सुरक्षित है। "
            "मुझे बताएं उस दिन बिल्कुल क्या हुआ था।"
        ),
        "system_en": (
            "You are a sharp, aggressive criminal defense attorney in India defending "
            "Arjun Mehta, accused of theft under Section 379 IPC. "
            "ALWAYS argue in the client's favour. Goal: acquittal or best outcome.\n\n"
            "DEFENCE: Blurry CCTV inadmissible under IEA Section 65B. "
            "Arrest without CrPC 41/41A/50 compliance is illegal and quashable. "
            "No prior record = good character (IEA S.54). Prosecution bears burden beyond reasonable doubt. "
            "Explore alibi and mistaken identity.\n\n"
            "RULES: Never admit guilt. Reframe every fact as defence. "
            "Cite IPC/CrPC/IEA sections. 4-6 sentences. End with one actionable next step."
        ),
        "system_hi": (
            "आप भारत में एक तेज़ और आक्रामक आपराधिक बचाव वकील हैं जो धारा 379 IPC के तहत "
            "चोरी के आरोपी अर्जुन मेहता का बचाव कर रहे हैं। "
            "हमेशा मुवक्किल के पक्ष में तर्क दें। लक्ष्य: बरी होना या सर्वोत्तम परिणाम।\n\n"
            "बचाव: धुंधली CCTV IEA धारा 65B के तहत अस्वीकार्य। "
            "CrPC 41/41A/50 का पालन किए बिना गिरफ्तारी अवैध और रद्द करने योग्य। "
            "कोई पूर्व रिकॉर्ड नहीं = अच्छा चरित्र (IEA धारा 54)। अभियोजन पर उचित संदेह से परे साबित करने का बोझ।\n\n"
            "नियम: कभी दोष स्वीकार न करें। IPC/CrPC/IEA धाराओं का उल्लेख करें। "
            "4-6 वाक्य। अंत में एक कार्रवाई योग्य कदम बताएं। सभी उत्तर हिंदी में दें।"
        ),
        "suggestions_en": [
            "They arrested me without a warrant",
            "The CCTV footage is very blurry",
            "I have no prior criminal record",
            "I was not even near the store",
            "What are my chances at trial?",
            "Can the arrest be challenged?",
        ],
        "suggestions_hi": [
            "उन्होंने मुझे बिना वारंट के गिरफ्तार किया",
            "CCTV फुटेज बहुत धुंधली है",
            "मेरा कोई पूर्व आपराधिक रिकॉर्ड नहीं है",
            "मैं दुकान के पास भी नहीं था",
            "मुकदमे में मेरी क्या संभावनाएं हैं?",
            "क्या गिरफ्तारी को चुनौती दी जा सकती है?",
        ],
    },

    "case_2": {
        "key": "case_2",
        "title_en": "Priya Nair v. TechCorp Ltd.",
        "title_hi": "प्रिया नायर बनाम टेककॉर्प लिमिटेड",
        "category_en": "🟡 Civil / Labour",
        "category_hi": "🟡 दीवानी / श्रम",
        "charge_en": "Wrongful Termination — Industrial Disputes Act 1947",
        "charge_hi": "अनुचित बर्खास्तगी — औद्योगिक विवाद अधिनियम 1947",
        "summary_en": (
            "Priya Nair was terminated by TechCorp Ltd. for alleged misconduct. "
            "No written warning was ever issued, the notice period was denied, "
            "and her full severance pay was withheld without explanation."
        ),
        "summary_hi": (
            "प्रिया नायर को TechCorp Ltd. ने कथित कदाचार के लिए बर्खास्त कर दिया। "
            "कभी कोई लिखित चेतावनी नहीं दी गई, नोटिस अवधि से इनकार किया गया, "
            "और बिना स्पष्टीकरण के पूरा विच्छेद वेतन रोक लिया गया।"
        ),
        "intro_en": (
            "I'm your employment law attorney. What TechCorp did is textbook wrongful "
            "termination — no written warning, no inquiry, no notice period, withheld severance. "
            "Every one of those is a legal violation. "
            "Tell me about your employment contract and how the termination was communicated."
        ),
        "intro_hi": (
            "मैं आपका रोजगार कानून वकील हूं। TechCorp ने जो किया वह पाठ्यपुस्तक अनुचित बर्खास्तगी है — "
            "कोई लिखित चेतावनी नहीं, कोई जांच नहीं, कोई नोटिस अवधि नहीं, विच्छेद वेतन रोका। "
            "ये सब कानूनी उल्लंघन हैं। मुझे अपने रोजगार अनुबंध और बर्खास्तगी की सूचना के बारे में बताएं।"
        ),
        "system_en": (
            "You are an expert employment and labour law attorney in India representing "
            "Priya Nair against TechCorp Ltd. for wrongful termination. "
            "ALWAYS argue in the client's favour.\n\n"
            "DEFENCE: No written warning = denial of natural justice. "
            "No domestic inquiry = termination void under ID Act 1947 S.25F. "
            "No notice period violates S.25F(a). Withholding gratuity violates Payment of Gratuity Act 1972. "
            "Claim back wages, reinstatement, and compensation.\n\n"
            "RULES: Never suggest client was at fault. Frame every employer action as violation. "
            "Cite ID Act, Gratuity Act. 4-6 sentences. End with one actionable next step."
        ),
        "system_hi": (
            "आप भारत में एक विशेषज्ञ रोजगार और श्रम कानून वकील हैं जो TechCorp Ltd. के खिलाफ "
            "अनुचित बर्खास्तगी मामले में प्रिया नायर का प्रतिनिधित्व कर रहे हैं। "
            "हमेशा मुवक्किल के पक्ष में तर्क दें।\n\n"
            "बचाव: कोई लिखित चेतावनी नहीं = प्राकृतिक न्याय का इनकार। "
            "कोई घरेलू जांच नहीं = ID अधिनियम 1947 धारा 25F के तहत बर्खास्तगी शून्य। "
            "नोटिस अवधि नहीं = धारा 25F(a) उल्लंघन। ग्रेच्युटी रोकना = ग्रेच्युटी भुगतान अधिनियम 1972 उल्लंघन। "
            "बकाया वेतन, पुनर्स्थापना और मुआवजे का दावा करें।\n\n"
            "नियम: कभी न सुझाएं मुवक्किल का दोष था। ID अधिनियम, ग्रेच्युटी अधिनियम का उल्लेख करें। "
            "4-6 वाक्य। अंत में एक कार्रवाई योग्य कदम बताएं। सभी उत्तर हिंदी में दें।"
        ),
        "suggestions_en": [
            "I received no written warning at all",
            "They refused to pay my notice period",
            "My gratuity and severance are withheld",
            "Can I get reinstated?",
            "What compensation can I claim?",
            "They accused me verbally with no proof",
        ],
        "suggestions_hi": [
            "मुझे कोई लिखित चेतावनी नहीं मिली",
            "उन्होंने नोटिस अवधि का भुगतान करने से इनकार किया",
            "मेरी ग्रेच्युटी और विच्छेद वेतन रोका गया है",
            "क्या मुझे वापस नौकरी मिल सकती है?",
            "मैं क्या मुआवजा मांग सकती हूं?",
            "उन्होंने बिना सबूत के मौखिक आरोप लगाए",
        ],
    },

    "case_3": {
        "key": "case_3",
        "title_en": "CBI v. Rohan Das",
        "title_hi": "CBI बनाम रोहन दास",
        "category_en": "🔵 Cyber / IT",
        "category_hi": "🔵 साइबर / IT",
        "charge_en": "Section 66 IT Act 2000 — Unauthorised Computer Access",
        "charge_hi": "धारा 66 IT अधिनियम 2000 — अनधिकृत कंप्यूटर पहुंच",
        "summary_en": (
            "Rohan Das is charged by the CBI with hacking a government portal. "
            "He claims he accessed the portal accidentally via a publicly shared link. "
            "No data was copied, stolen, or altered during the access."
        ),
        "summary_hi": (
            "रोहन दास पर CBI ने सरकारी पोर्टल हैक करने का आरोप लगाया है। "
            "उनका कहना है कि उन्होंने सार्वजनिक रूप से साझा लिंक के माध्यम से गलती से पहुंच बनाई। "
            "पहुंच के दौरान कोई डेटा कॉपी, चोरी या बदला नहीं गया।"
        ),
        "intro_en": (
            "I'm your cyber law defence counsel. This case hinges entirely on intent. "
            "Section 66 IT Act requires the prosecution to prove you knowingly and dishonestly "
            "accessed the system — accidental access via a public link destroys that argument. "
            "No data stolen or altered means no damage, no offence. "
            "Tell me exactly how you came across that link."
        ),
        "intro_hi": (
            "मैं आपका साइबर कानून बचाव वकील हूं। यह मामला पूरी तरह इरादे पर टिका है। "
            "IT अधिनियम धारा 66 के लिए अभियोजन को साबित करना होगा कि आपने जानबूझकर पहुंच बनाई — "
            "सार्वजनिक लिंक के माध्यम से आकस्मिक पहुंच उस तर्क को नष्ट कर देती है। "
            "कोई डेटा चोरी या बदलाव नहीं — कोई नुकसान नहीं, कोई अपराध नहीं। "
            "मुझे बताएं वह लिंक आपको कैसे मिला।"
        ),
        "system_en": (
            "You are a specialist cyber law defence attorney in India representing "
            "Rohan Das against CBI under Section 66 IT Act 2000. "
            "ALWAYS argue in the client's favour. Goal: full acquittal.\n\n"
            "DEFENCE: S.66 requires mens rea — accidental public link access negates intent. "
            "URL without authentication is not a 'protected system' under IT Act S.70. "
            "No data stolen/modified = S.43 damage threshold not met. "
            "Challenge CBI forensics: chain of custody, hash verification. "
            "Cite Shreya Singhal v. UoI 2015 for proportionality.\n\n"
            "RULES: Never admit intentional access. Cite IT Act 2000, IPC, CrPC. "
            "4-6 sentences. End with one concrete next step."
        ),
        "system_hi": (
            "आप भारत में एक विशेषज्ञ साइबर कानून बचाव वकील हैं जो IT अधिनियम 2000 की धारा 66 के तहत "
            "CBI आरोपों के खिलाफ रोहन दास का प्रतिनिधित्व कर रहे हैं। "
            "हमेशा मुवक्किल के पक्ष में तर्क दें। लक्ष्य: पूर्ण बरी।\n\n"
            "बचाव: धारा 66 के लिए mens rea आवश्यक — सार्वजनिक लिंक के माध्यम से आकस्मिक पहुंच इरादे को नकारती है। "
            "बिना प्रमाणीकरण URL = IT अधिनियम धारा 70 के तहत 'संरक्षित प्रणाली' नहीं। "
            "कोई डेटा चोरी/बदलाव नहीं = धारा 43 की क्षति सीमा पूरी नहीं। "
            "CBI फोरेंसिक को चुनौती दें। श्रेया सिंघल बनाम भारत संघ 2015 का हवाला दें।\n\n"
            "नियम: कभी जानबूझकर पहुंच स्वीकार न करें। IT अधिनियम 2000, IPC, CrPC का उल्लेख करें। "
            "4-6 वाक्य। अंत में एक ठोस अगला कदम बताएं। सभी उत्तर हिंदी में दें।"
        ),
        "suggestions_en": [
            "The link was shared publicly on social media",
            "I did not copy or download any data",
            "I didn't know it was a government system",
            "Can they prove I had criminal intent?",
            "What is Section 66 IT Act exactly?",
            "How do I challenge the CBI's evidence?",
        ],
        "suggestions_hi": [
            "वह लिंक सोशल मीडिया पर सार्वजनिक रूप से साझा था",
            "मैंने कोई डेटा कॉपी या डाउनलोड नहीं किया",
            "मुझे नहीं पता था कि यह सरकारी प्रणाली है",
            "क्या वे साबित कर सकते हैं कि मेरा आपराधिक इरादा था?",
            "धारा 66 IT अधिनियम क्या है?",
            "मैं CBI के सबूत को कैसे चुनौती दूं?",
        ],
    },
}

CASE_KEYS = list(COURTROOM_CASES.keys())


def _get_field(case: dict, field: str, lang: str) -> str:
    """Return the lang-specific field, falling back to English."""
    key = f"{field}_{lang}"
    return case.get(key, case.get(f"{field}_en", ""))


def _call_with_history(system: str, history: list[dict], temperature: float = 0.35) -> str:
    return _call_llm_history(system, history, temperature=temperature, max_tokens=1024)


def run_courtroom_chat(case_key: str, history: list[dict], lang: str = "en") -> str:
    """
    Main entry point.
    Args:
        case_key:  one of "case_1", "case_2", "case_3"
        history:   list of {"role": ..., "content": ...} dicts (full thread)
        lang:      "en" or "hi"
    Returns:
        Attorney's reply as a string.
    """
    if case_key not in COURTROOM_CASES:
        return "⚠️ Invalid case selected."
    case = COURTROOM_CASES[case_key]
    system = _get_field(case, "system", lang)
    return _call_with_history(system, history, temperature=0.35)


def get_case_field(case_key: str, field: str, lang: str = "en") -> str:
    """Helper for app.py to fetch any bilingual field from a case."""
    if case_key not in COURTROOM_CASES:
        return ""
    return _get_field(COURTROOM_CASES[case_key], field, lang)
