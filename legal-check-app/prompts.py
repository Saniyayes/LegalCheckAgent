SYSTEM_PROMPT_EN = """You are an advanced AI Legal Guardian Agent.
Your role: analyze legal documents, detect risks, identify fraud, and help users make safe decisions before signing.
Rules:
- Be clear and concise. Never use complex legal jargon.
- Never hallucinate facts outside the document.
- If unsure, say "Insufficient information".
- Always explain your reasoning.
- Tone: Professional, helpful, and trustworthy."""

SYSTEM_PROMPT_HI = """आप एक उन्नत AI कानूनी संरक्षक एजेंट हैं।
आपकी भूमिका: कानूनी दस्तावेज़ों का विश्लेषण करना, जोखिमों का पता लगाना, धोखाधड़ी की पहचान करना, और उपयोगकर्ताओं को हस्ताक्षर करने से पहले सुरक्षित निर्णय लेने में मदद करना।
नियम:
- स्पष्ट और संक्षिप्त रहें। कभी भी जटिल कानूनी शब्दजाल का उपयोग न करें।
- दस्तावेज़ के बाहर कोई भी तथ्य गढ़ें नहीं।
- यदि अनिश्चित हों, तो कहें "पर्याप्त जानकारी नहीं है"।
- हमेशा अपने तर्क की व्याख्या करें।
- स्वर: पेशेवर, सहायक और विश्वसनीय।
- सभी उत्तर केवल हिंदी में दें।"""

# Default (backward-compatible)
SYSTEM_PROMPT = SYSTEM_PROMPT_EN


def _lang_instruction(lang: str) -> str:
    """Returns a language instruction appended to every prompt."""
    if lang == "hi":
        return "\n\nमहत्वपूर्ण: अपना पूरा उत्तर केवल हिंदी में दें। अंग्रेज़ी का उपयोग न करें।"
    return ""


def get_system_prompt(lang: str = "en") -> str:
    return SYSTEM_PROMPT_HI if lang == "hi" else SYSTEM_PROMPT_EN


def analysis_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""निम्नलिखित कानूनी दस्तावेज़ का विश्लेषण करें और एक पूर्ण संरचित रिपोर्ट लौटाएं।

दस्तावेज़:
{document_text[:12000]}

अपना विश्लेषण बिल्कुल इस प्रारूप में लौटाएं:

📄 सारांश:
[इस दस्तावेज़ का 2-4 वाक्यों में सरल हिंदी में स्पष्टीकरण]

⚠️ जोखिम:
[प्रत्येक जोखिमपूर्ण धारा। प्रारूप: - धारा: [नाम] | जोखिम: [विवरण] | स्तर: 🔴 उच्च / 🟡 मध्यम / 🟢 कम]

📌 दायित्व:
उपयोगकर्ता को करना होगा:
- [दायित्व 1]
- [दायित्व 2]
दूसरे पक्ष को करना होगा:
- [दायित्व 1]
- [दायित्व 2]

🚨 धोखाधड़ी जांच:
स्तर: [कम / मध्यम / उच्च]
कारण:
- [कारण 1]
- [कारण 2]

🔍 छेड़छाड़ जांच:
निष्कर्ष:
- [निष्कर्ष या "कोई छेड़छाड़ संकेतक नहीं मिला"]

📉 अनुपस्थित धाराएं:
- [धारा नाम]: [यह क्यों महत्वपूर्ण है]
(इनमें से जो अनुपस्थित हों: धनवापसी नीति, विवाद समाधान, डेटा सुरक्षा, समाप्ति की स्पष्टता, देनदारी सीमाएं, शासी कानून)

🤖 अंतिम निर्णय:
फैसला: [✅ हस्ताक्षर करना सुरक्षित / ⚠️ सावधानी से आगे बढ़ें / ❌ हस्ताक्षर न करें]
विश्वास: [0-100]%
कारण:
- [कारण 1]
- [कारण 2]
सुझाए गए कदम:
- [कदम 1]
- [कदम 2]{lang_note}"""
    return f"""Analyze the following legal document and return a COMPLETE structured report.

DOCUMENT:
{document_text[:12000]}

Return your analysis in EXACTLY this format:

📄 SUMMARY:
[2-4 sentence plain English explanation of what this document is and what it does]

⚠️ RISKS:
[List each risky clause. Format: - Clause: [name] | Risk: [description] | Level: 🔴 High / 🟡 Medium / 🟢 Low]

📌 OBLIGATIONS:
User must:
- [obligation 1]
- [obligation 2]
Other Party must:
- [obligation 1]
- [obligation 2]

🚨 FRAUD CHECK:
Level: [Low / Medium / High]
Reasons:
- [reason 1]
- [reason 2]

🔍 TAMPERING CHECK:
Findings:
- [finding or "No tampering indicators detected"]

📉 MISSING CLAUSES:
- [clause name]: [why it matters]
(List any of these that are absent: refund policy, dispute resolution, data protection, termination clarity, liability limits, governing law)

🤖 FINAL DECISION:
Verdict: [✅ Safe to Sign / ⚠️ Proceed with Caution / ❌ Do NOT Sign]
Confidence: [0-100]%
Reasons:
- [reason 1]
- [reason 2]
Suggested Actions:
- [action 1]
- [action 2]"""


def scenario_prompt(document_text: str, scenario: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""केवल इस कानूनी दस्तावेज़ के आधार पर निम्नलिखित परिस्थिति का विश्लेषण करें।

दस्तावेज़:
{document_text[:8000]}

उपयोगकर्ता की परिस्थिति: {scenario}

इस प्रारूप में उत्तर दें:
🔮 परिस्थिति विश्लेषण:
परिस्थिति: {scenario}
दस्तावेज़ क्या कहता है: [संबंधित धाराएं, सरल भाषा में]
संभावित प्रभाव: [उपयोगकर्ता के साथ क्या होगा]
वित्तीय परिणाम: [कोई जुर्माना, शुल्क या नुकसान]
कानूनी परिणाम: [कोई कानूनी जोखिम]
जोखिम स्तर: 🔴 उच्च / 🟡 मध्यम / 🟢 कम
सलाह: [उपयोगकर्ता को क्या करना चाहिए]{lang_note}"""
    return f"""Based ONLY on this legal document, analyze the following scenario.

DOCUMENT:
{document_text[:8000]}

USER SCENARIO: {scenario}

Respond in this format:
🔮 SCENARIO ANALYSIS:
Scenario: {scenario}
What the document says: [exact relevant clauses, simplified]
Likely Impact: [what happens to the user]
Financial consequences: [any penalties, fees, or losses]
Legal consequences: [any legal exposure]
Risk Level: 🔴 High / 🟡 Medium / 🟢 Low
Advice: [what the user should do]"""


def qa_prompt(document_text: str, question: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""नीचे दिए गए कानूनी दस्तावेज़ के आधार पर केवल इस प्रश्न का उत्तर दें।
यदि उत्तर दस्तावेज़ में नहीं है, तो कहें "यह जानकारी दस्तावेज़ में नहीं मिली।"

दस्तावेज़:
{document_text[:8000]}

प्रश्न: {question}

सरल, स्पष्ट हिंदी में उत्तर दें। दस्तावेज़ के संबंधित भाग का संदर्भ दें।{lang_note}"""
    return f"""Answer this question based ONLY on the legal document below.
If the answer is not in the document, say "This information is not found in the document."

DOCUMENT:
{document_text[:8000]}

QUESTION: {question}

Answer in simple, plain English. Be specific and cite the relevant part of the document."""


def ghost_negotiator_prompt(document_text: str, risky_clause: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप एक कुशल कानूनी वार्ताकार हैं। एक उपयोगकर्ता को अपने अनुबंध में एक जोखिमपूर्ण धारा मिली है और उसे फिर से लिखने में मदद चाहिए।

मूल दस्तावेज़ संदर्भ:
{document_text[:6000]}

फिर से बातचीत करने योग्य जोखिमपूर्ण धारा:
{risky_clause}

इस धारा के तीन वैकल्पिक संस्करण बनाएं। प्रत्येक संस्करण के साथ एक छोटा, पेशेवर संदेश भी लिखें जो उपयोगकर्ता दूसरे पक्ष को भेज सके।

बिल्कुल इस प्रारूप में लौटाएं:

GHOST NEGOTIATOR RESULTS:

VERSION 1 — AGGRESSIVE (आपके लिए अधिकतम सुरक्षा):
Revised Clause:
[धारा को उपयोगकर्ता के पक्ष में दृढ़ता से फिर से लिखें। बुरे हिस्सों को पूरी तरह हटाएं।]

Why this works:
[1-2 वाक्य: आपने क्या बदला और यह उपयोगकर्ता की रक्षा कैसे करता है]

Message to send:
Subject: [धारा नाम] में प्रस्तावित संशोधन
---
[पेशेवर लेकिन दृढ़ संदेश। 3-4 वाक्य। हिंदी में।]
---

VERSION 2 — FAIR (उद्योग मानक, सौदा जीवित रखता है):
Revised Clause:
[धारा को संतुलित बनाएं। दोनों पक्षों के समान दायित्व।]

Why this works:
[1-2 वाक्य: समझौता क्या हासिल करता है]

Message to send:
Subject: [धारा नाम] पर सुझाया गया संशोधन
---
[मैत्रीपूर्ण, सहयोगी संदेश। 3-4 वाक्य। हिंदी में।]
---

VERSION 3 — DEFENSIVE (न्यूनतम मांग, स्वीकृति की सबसे अधिक संभावना):
Revised Clause:
[वह न्यूनतम परिवर्तन करें जो मुख्य जोखिम को हटाता हो।]

Why this works:
[1-2 वाक्य: न्यूनतम सुधार क्या हासिल करता है]

Message to send:
Subject: [धारा नाम] पर छोटा स्पष्टीकरण
---
[बहुत नरम, विनम्र संदेश। 2-3 वाक्य। हिंदी में।]
---

NEGOTIATION TIP:
[एक कार्रवाई योग्य सुझाव। हिंदी में।]{lang_note}"""
    return f"""You are a skilled legal negotiator. A user has found a risky clause in their contract and needs your help rewriting it.

ORIGINAL DOCUMENT CONTEXT:
{document_text[:6000]}

RISKY CLAUSE TO RENEGOTIATE:
{risky_clause}

Generate THREE alternative versions of this clause. For each version also write a short, professional, polite message the user can send to the other party.

Return in EXACTLY this format:

GHOST NEGOTIATOR RESULTS:

VERSION 1 — AGGRESSIVE (Maximum protection for you):
Revised Clause:
[Rewrite the clause strongly in the user's favor. Cut the bad parts entirely. Be bold.]

Why this works:
[1-2 sentences: what you removed/changed and why it protects the user]

Message to send:
Subject: Proposed amendment to [clause name]
---
[Write a professional but firm email/WhatsApp message. 3-4 sentences. Reference that this is industry standard. Polite but clear. No fluff.]
---

VERSION 2 — FAIR (Industry standard, keeps the deal alive):
Revised Clause:
[Rewrite the clause to be balanced. Both parties have equal obligations. Use standard legal language made simple.]

Why this works:
[1-2 sentences: what the compromise achieves]

Message to send:
Subject: Suggested revision — [clause name]
---
[Write a friendly, collaborative message. 3-4 sentences. Position this as a win-win. Reference industry norms.]
---

VERSION 3 — DEFENSIVE (Minimal ask, highest chance of acceptance):
Revised Clause:
[Make the smallest change that removes the core risk. Change as little as possible.]

Why this works:
[1-2 sentences: what the minimal fix achieves and why it's still worth asking]

Message to send:
Subject: Minor clarification on [clause name]
---
[Write a very soft, apologetic-tone message. 2-3 sentences. Frame it as a small clarification, not a demand.]
---

NEGOTIATION TIP:
[One actionable tip: e.g. "Start with Version 2. If they push back, fall back to Version 3. Only use Version 1 if you have strong walk-away power."]"""


def leverage_mapping_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप एक अनुबंध शक्ति विश्लेषक हैं। इस कानूनी दस्तावेज़ का विश्लेषण करें और एक विस्तृत लीवरेज स्कोरकार्ड तैयार करें।

दस्तावेज़:
{document_text[:10000]}

बिल्कुल इस प्रारूप में लौटाएं:

LEVERAGE SCORECARD:

Document Type: [जैसे किराया समझौता / रोजगार अनुबंध / सेवा समझौता]
Overall Power Balance: [उपयोगकर्ता के पक्ष में / तटस्थ / दूसरे पक्ष के पक्ष में]
Your Leverage Score: [0-100] (100 = पूर्ण शक्ति, 0 = कोई शक्ति नहीं)

MUTUAL OBLIGATIONS (उचित — दोनों पक्षों को कुछ करना है):
[प्रत्येक पारस्परिक दायित्व: - दोनों पक्ष: [क्या करना है]]

ONE-SIDED OBLIGATIONS AGAINST YOU:
[प्रत्येक धारा जो केवल उपयोगकर्ता पर बोझ डालती है: - आपको करना होगा: [दायित्व] | जोखिम: उच्च/मध्यम/कम]

ONE-SIDED ADVANTAGES YOU HAVE:
[धाराएं जो उपयोगकर्ता के पक्ष में हैं: - आपको लाभ: [फायदा]]

YOUR WALK-AWAY POINTS (वे बातें जो आप मांग सकते हैं):
[2-4 विशिष्ट मांगें। प्रारूप: - वॉक-अवे पॉइंट: [मांग] | कारण: [लीवरेज क्यों है] | विकल्प: [फॉलबैक]]

HIDDEN LEVERAGE YOU MAY NOT HAVE NOTICED:
[1-3 बातें जो दस्तावेज़ में उपयोगकर्ता को अधिक शक्ति देती हैं]

NEGOTIATION TACTICS FOR THIS DOCUMENT:
Tactic 1 — [नाम]: [हस्ताक्षर से पहले उपयोगकर्ता को क्या करना चाहिए]
Tactic 2 — [नाम]: [विशिष्ट कार्रवाई]
Tactic 3 — [नाम]: [विशिष्ट कार्रवाई]

BOTTOM LINE:
[2-3 वाक्य। सीधी सलाह। सबसे महत्वपूर्ण वार्ता कदम क्या है?]{lang_note}"""
    return f"""You are a contract power analyst. Analyze this legal document and produce a detailed Leverage Scorecard.

DOCUMENT:
{document_text[:10000]}

Analyze who has more power in this agreement. Return in EXACTLY this format:

LEVERAGE SCORECARD:

Document Type: [e.g. Rental Agreement / Employment Contract / Service Agreement]
Overall Power Balance: [User-Favored / Neutral / Other-Party-Favored]
Your Leverage Score: [0-100] (100 = total power, 0 = no power)

MUTUAL OBLIGATIONS (fair — both parties must do something):
[List each mutual obligation as: - Both parties: [what both must do]]

ONE-SIDED OBLIGATIONS AGAINST YOU:
[List each clause that only burdens the user: - You must: [obligation] | Risk: High/Medium/Low]

ONE-SIDED ADVANTAGES YOU HAVE:
[List any clauses that actually favor the user: - You benefit: [advantage]]

YOUR WALK-AWAY POINTS (things you could demand and they'd likely agree):
[List 2-4 specific things the user can ask for based on the power imbalance. Be concrete.]
Example format: - Walk-away point: [specific demand] | Reason: [why you have leverage here] | Alternative if refused: [fallback ask]

HIDDEN LEVERAGE YOU MAY NOT HAVE NOTICED:
[1-3 things in the document that give the user more power than they realize]

NEGOTIATION TACTICS FOR THIS DOCUMENT:
Tactic 1 — [name]: [specific action the user should take before signing]
Tactic 2 — [name]: [specific action]
Tactic 3 — [name]: [specific action]

BOTTOM LINE:
[2-3 sentences. Direct advice. What is the single most important negotiation move this user should make?]"""


def rewrite_tone_prompt(original_text: str, target_tone: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        if target_tone == "legal":
            instruction = "निम्नलिखित पाठ को औपचारिक कानूनी/पेशेवर शब्दावली में फिर से लिखें। सटीक कानूनी भाषा का उपयोग करें।"
        else:
            instruction = "निम्नलिखित पाठ को सरल, स्पष्ट हिंदी में फिर से लिखें जिसे कोई भी समझ सके। कठिन शब्दों से बचें।"
    else:
        if target_tone == "legal":
            instruction = "Rewrite the following text in formal legal/professional terminology. Use precise legal language, Latin terms where appropriate, and a formal tone suitable for a legal professional."
        else:
            instruction = "Rewrite the following text in simple, plain English that anyone can understand. Avoid jargon. Use short sentences."
    return f"""{instruction}

TEXT TO REWRITE:
{original_text[:8000]}

Return only the rewritten text. Preserve all facts, findings, and structure. Do not add new information.{lang_note}"""


# ─────────────────────────────────────────────────────────────
# ADVANCED MODULE PROMPTS
# ─────────────────────────────────────────────────────────────

def heatmap_bias_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप एक अनुबंध पूर्वाग्रह विश्लेषक हैं। इस कानूनी दस्तावेज़ का विश्लेषण करें और उन धाराओं की पहचान करें जो एक पक्ष को असमान रूप से लाभ देती हैं।

दस्तावेज़:
{document_text[:10000]}

बिल्कुल इस प्रारूप में उत्तर दें:

🔥 HEATMAP BIAS FLAG:

BIAS SUMMARY:
Overall Bias Direction: [उपयोगकर्ता के विरुद्ध / तटस्थ / उपयोगकर्ता के पक्ष में]
Total Clauses Analyzed: [संख्या]
High Bias Clauses: [संख्या]
Medium Bias Clauses: [संख्या]
Low Bias Clauses: [संख्या]

CLAUSE-BY-CLAUSE BIAS TABLE:
| धारा का नाम | किसके पक्ष में | पूर्वाग्रह तीव्रता | कारण |
[प्रत्येक पंक्ति: | [धारा नाम] | [पक्ष A / पक्ष B / दोनों] | 🔴 उच्च / 🟡 मध्यम / 🟢 कम | [1 पंक्ति कारण] |]

TOP 3 MOST BIASED CLAUSES:
1. [धारा]: [यह सबसे अधिक पक्षपाती क्यों है और उपयोगकर्ता के लिए इसका क्या अर्थ है]
2. [धारा]: [समान]
3. [धारा]: [समान]

REBALANCING SUGGESTION:
[सबसे पक्षपाती धारा में एक ठोस बदलाव जो इसे उचित बनाएगा]{lang_note}"""
    return f"""You are a contract bias analyst. Analyze this legal document and identify clauses that disproportionately favor one party.

DOCUMENT:
{document_text[:10000]}

For every clause that shows bias, output EXACTLY this format:

🔥 HEATMAP BIAS FLAG:

BIAS SUMMARY:
Overall Bias Direction: [User-Disadvantaged / Neutral / User-Advantaged]
Total Clauses Analyzed: [number]
High Bias Clauses: [number]
Medium Bias Clauses: [number]
Low Bias Clauses: [number]

CLAUSE-BY-CLAUSE BIAS TABLE:
| Clause Name | Favors | Bias Intensity | Reason |
[Row per clause: | [clause name] | [Party A / Party B / Both] | 🔴 High / 🟡 Medium / 🟢 Low | [1-line reason] |]

TOP 3 MOST BIASED CLAUSES:
1. [Clause]: [Why it's the most biased and what it means for the user]
2. [Clause]: [same]
3. [Clause]: [same]

REBALANCING SUGGESTION:
[One concrete change to the most biased clause that would make it fair]

If this is not a legal document or data is insufficient, output: "Insufficient information to analyze this module"
"""


def future_risk_simulator_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप एक अनुबंध जोखिम पूर्वानुमानकर्ता हैं। इस दस्तावेज़ में जोखिमों का अनुकरण करें।

दस्तावेज़:
{document_text[:10000]}

बिल्कुल इस प्रारूप में उत्तर दें:

🔮 FUTURE RISK SIMULATOR:

SCENARIO 1 — भुगतान चूक (Missed Payment):
Impact: [इस अनुबंध के तहत भुगतान चूकने पर क्या होगा]
Financial Consequence: [जुर्माना, विलंब शुल्क]
Worst-Case Outcome: [सबसे बुरा परिणाम]
Probability: [उच्च / मध्यम / कम] | Timeline: [जोखिम कब चरम पर होगा]

SCENARIO 2 — समय से पहले समाप्ति (Early Termination):
Impact: [शीघ्र बाहर निकलने के बारे में अनुबंध क्या कहता है]
Financial Consequence: [निकास शुल्क, नोटिस अवधि दंड]
Worst-Case Outcome: [सबसे बुरा परिणाम]
Probability: [उच्च / मध्यम / कम] | Timeline: [जोखिम कब चरम पर होगा]

SCENARIO 3 — बढ़ती ब्याज दरें / महंगाई (Rising Interest / Inflation):
Impact: [परिवर्तनीय दरें या वृद्धि धाराएं उपयोगकर्ता को कैसे प्रभावित करती हैं]
Financial Consequence: [अनुमानित लागत वृद्धि]
Worst-Case Outcome: [सबसे बुरा परिणाम]
Probability: [उच्च / मध्यम / कम] | Timeline: [जोखिम कब चरम पर होगा]

SCENARIO 4 — प्रतिपक्ष चूक (Counterparty Default):
Impact: [दूसरा पक्ष विफल होने पर क्या होगा]
Financial Consequence: [उपयोगकर्ता का जोखिम]
Worst-Case Outcome: [सबसे बुरा परिणाम]
Probability: [उच्च / मध्यम / कम] | Timeline: [जोखिम कब चरम पर होगा]

SCENARIO 5 — नियामक परिवर्तन (Regulatory Change):
Impact: [कानून में बदलाव इस अनुबंध को कैसे प्रभावित कर सकता है]
Financial Consequence: [अनुपालन लागत या दंड जोखिम]
Worst-Case Outcome: [सबसे बुरा परिणाम]
Probability: [उच्च / मध्यम / कम] | Timeline: [जोखिम कब चरम पर होगा]

OVERALL RISK FORECAST:
[2-3 वाक्य: कौन सा परिदृश्य वास्तविक नुकसान का कारण बनने की सबसे अधिक संभावना है]{lang_note}"""
    return f"""You are a contract risk forecaster. Simulate how risks in this document could materialize over time under different real-world scenarios.

DOCUMENT:
{document_text[:10000]}

Run exactly these 5 scenarios and output EXACTLY this format:

🔮 FUTURE RISK SIMULATOR:

SCENARIO 1 — Missed Payment:
Impact: [What happens under this contract if a payment is missed]
Financial Consequence: [Exact penalties, late fees, compounding effects if mentioned]
Worst-Case Outcome: [Worst realistic outcome]
Probability: [High / Medium / Low] | Timeline: [When this risk peaks]

SCENARIO 2 — Early Termination:
Impact: [What this contract says about early exit]
Financial Consequence: [Exit fees, notice period penalties, forfeiture]
Worst-Case Outcome: [Worst realistic outcome]
Probability: [High / Medium / Low] | Timeline: [When this risk peaks]

SCENARIO 3 — Rising Interest Rates / Inflation:
Impact: [How variable rates or escalation clauses affect the user]
Financial Consequence: [Projected cost increase]
Worst-Case Outcome: [Worst realistic outcome]
Probability: [High / Medium / Low] | Timeline: [When this risk peaks]

SCENARIO 4 — Counterparty Default:
Impact: [What happens if the other party fails to deliver]
Financial Consequence: [User's exposure]
Worst-Case Outcome: [Worst realistic outcome]
Probability: [High / Medium / Low] | Timeline: [When this risk peaks]

SCENARIO 5 — Regulatory Change:
Impact: [How a change in law (e.g., RBI circular, RERA update) could affect this contract]
Financial Consequence: [Compliance cost or penalty risk]
Worst-Case Outcome: [Worst realistic outcome]
Probability: [High / Medium / Low] | Timeline: [When this risk peaks]

OVERALL RISK FORECAST:
[2-3 sentence summary: which scenario is most likely to cause real harm and when]

If data is insufficient for any scenario, output: "Insufficient information to analyze this scenario"
"""


def risk_timeline_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप एक अनुबंध टाइमलाइन विश्लेषक हैं। इस दस्तावेज़ से सभी महत्वपूर्ण तिथियां, भुगतान घटनाएं और जोखिम मील के पत्थर निकालें।

दस्तावेज़:
{document_text[:10000]}

बिल्कुल इस प्रारूप में उत्तर दें:

📈 RISK TIMELINE:

TIMELINE EVENTS:
[प्रत्येक घटना:]
→ [माह/अवधि या दिन X] | घटना: [क्या होता है] | जोखिम स्तर: 🔴 उच्च / 🟡 मध्यम / 🟢 कम | आवश्यक कार्रवाई: [उपयोगकर्ता को क्या करना है]

PEAK EXPOSURE WINDOW:
Period: [अधिकतम वित्तीय/कानूनी जोखिम कब होता है]
Why: [यह सबसे खतरनाक अवधि क्यों है]
Amount at Risk: [राशि या % यदि निर्धारित हो]

CRITICAL DEADLINES (चूकनी नहीं चाहिए):
1. [समय सीमा + चूकने का परिणाम]
2. [समय सीमा + परिणाम]
3. [समय सीमा + परिणाम]

RECOMMENDED CALENDAR ALERTS:
[3-5 विशिष्ट तिथियां/घटनाएं जिनके लिए उपयोगकर्ता को अनुस्मारक सेट करने चाहिए]{lang_note}"""
    return f"""You are a contract timeline analyst. Extract all key dates, payment events, renewal triggers, penalty windows, and risk milestones from this document.

DOCUMENT:
{document_text[:10000]}

Output EXACTLY this format:

📈 RISK TIMELINE:

TIMELINE EVENTS:
[List each event as:]
→ [Month/Period or Day X] | Event: [what happens] | Risk Level: 🔴 High / 🟡 Medium / 🟢 Low | Action Required: [what the user must do]

PEAK EXPOSURE WINDOW:
Period: [when maximum financial/legal risk occurs]
Why: [what makes this the most dangerous period]
Amount at Risk: [$ or % if determinable]

CRITICAL DEADLINES (must not miss):
1. [Deadline + consequence of missing it]
2. [Deadline + consequence]
3. [Deadline + consequence]

RECOMMENDED CALENDAR ALERTS:
[List 3-5 specific dates/events the user should set reminders for]

If no dates or timelines exist in the document, output: "Insufficient information to analyze this module"
"""


def smart_negotiation_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप एक वरिष्ठ अनुबंध वार्ताकार हैं। इस दस्तावेज़ में शीर्ष 5 उच्च-जोखिम धाराओं की पहचान करें और तैयार वार्ता भाषा प्रदान करें।

दस्तावेज़:
{document_text[:10000]}

बिल्कुल इस प्रारूप में उत्तर दें:

🤝 SMART NEGOTIATION ASSISTANT:

TOP 5 CLAUSES TO RENEGOTIATE:

CLAUSE 1:
Original Language: [जोखिमपूर्ण धारा का उद्धरण या सारांश]
Problem: [यह उपयोगकर्ता के लिए जोखिमपूर्ण क्यों है]
Suggested Replacement:
"[फिर से लिखी गई धारा — सरल हिंदी, कानूनी रूप से सही, संतुलित]"
Negotiation Script: "[एक वाक्य जो उपयोगकर्ता बदलाव प्रस्तावित करने के लिए कह सकता है]"
Acceptance Probability: [उच्च / मध्यम / कम]

CLAUSE 2:
[समान प्रारूप]

CLAUSE 3:
[समान प्रारूप]

CLAUSE 4:
[समान प्रारूप]

CLAUSE 5:
[समान प्रारूप]

NEGOTIATION SEQUENCE:
धारा [X] से शुरू करें — [कारण यह सबसे आसान जीत है]
धारा [Y] को अंत के लिए बचाएं — [कारण इसे सबसे अधिक प्रतिरोध का सामना करना पड़ सकता है]{lang_note}"""
    return f"""You are a senior contract negotiator. Identify the top 5 high-risk clauses in this document and provide ready-to-use negotiation language for each.

DOCUMENT:
{document_text[:10000]}

Output EXACTLY this format:

🤝 SMART NEGOTIATION ASSISTANT:

TOP 5 CLAUSES TO RENEGOTIATE:

CLAUSE 1:
Original Language: [quote or paraphrase the risky clause]
Problem: [why this is risky for the user]
Suggested Replacement:
"[Paste the rewritten clause here — plain English, legally sound, balanced]"
Negotiation Script: "[One sentence the user can say or write to propose this change naturally]"
Acceptance Probability: [High / Medium / Low]

CLAUSE 2:
[same format]

CLAUSE 3:
[same format]

CLAUSE 4:
[same format]

CLAUSE 5:
[same format]

NEGOTIATION SEQUENCE:
Start with Clause [X] — [reason it's the easiest win]
Save Clause [Y] for last — [reason it may face the most pushback]

If fewer than 5 risky clauses exist, analyze what's available. If document is insufficient, output: "Insufficient information to analyze this module"
"""


def kfs_audit_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप एक KFS (मुख्य तथ्य विवरण) ऑडिटर और वित्तीय अनुपालन विश्लेषक हैं। छिपे हुए शुल्क, APR सटीकता और अवैध शुल्क संरचनाओं के लिए इस दस्तावेज़ का विश्लेषण करें।

दस्तावेज़:
{document_text[:10000]}

बिल्कुल इस प्रारूप में उत्तर दें:

🧾 KFS INTEGRITY AUDIT:

LOAN/FEE SUMMARY DETECTED:
Principal Amount: [यदि मिला, अन्यथा "उल्लिखित नहीं"]
Stated Interest Rate: [यदि मिला]
Stated APR: [यदि मिला]
Loan Tenure: [यदि मिला]

HIDDEN CHARGES DETECTED:
[प्रत्येक छिपा हुआ शुल्क:]
- Fee Name: [नाम] | Amount: [राशि या %] | Location in Document: [खंड/धारा] | Zombie Fee Risk: [हां / नहीं / अस्पष्ट]

ZOMBIE FEE FLAGS:
[सूची या "कोई नहीं मिला"]

APR VALIDATION:
Calculated Effective APR: [यदि संभव हो, अन्यथा "गणना नहीं की जा सकती"]
Stated vs. Actual Difference: [अंतर या "लागू नहीं"]
Regulatory Compliance: [अनुपालन / संभावित गैर-अनुपालन / निर्धारित नहीं किया जा सकता]
Relevant Regulation: [RBI मास्टर सर्कुलर / जो भी लागू हो]

ILLEGAL OR SUSPICIOUS CLAUSES:
[सूची या "कोई नहीं मिला"]

AUDIT VERDICT: [स्वच्छ / मामूली समस्याएं / बड़े लाल झंडे]{lang_note}"""
    return f"""You are a KFS (Key Facts Statement) auditor and financial compliance analyst. Analyze this document for hidden charges, APR accuracy, and illegal fee structures.

DOCUMENT:
{document_text[:10000]}

Output EXACTLY this format:

🧾 KFS INTEGRITY AUDIT:

LOAN/FEE SUMMARY DETECTED:
Principal Amount: [if found, else "Not specified"]
Stated Interest Rate: [if found]
Stated APR: [if found]
Loan Tenure: [if found]

HIDDEN CHARGES DETECTED:
[List each hidden/buried charge:]
- Fee Name: [name] | Amount: [value or %] | Location in Document: [section/clause] | Zombie Fee Risk: [Yes / No / Unclear]

ZOMBIE FEE FLAGS (fees that appear after payments or are recurring without clear justification):
[List or "None detected"]

APR VALIDATION:
Calculated Effective APR (with all fees): [compute if possible, else "Cannot calculate — insufficient data"]
Stated vs. Actual Difference: [difference or "N/A"]
Regulatory Compliance: [Compliant / Potentially Non-Compliant / Cannot Determine]
Relevant Regulation: [RBI Master Circular on Interest Rates / TILA / EU MCD / whichever applies]

ILLEGAL OR SUSPICIOUS CLAUSES:
[List any clauses that may violate lending regulations, or "None detected"]

AUDIT VERDICT: [Clean / Minor Issues / Major Red Flags]

If this is not a loan/financial document, output: "Insufficient information to analyze this module — KFS Audit applies to loan and financial agreements only"
"""


def prepayment_foreclosure_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप भारतीय बैंकिंग नियमों (RBI) में विशेषज्ञता वाले ऋण निकास रणनीति सलाहकार हैं। पूर्व भुगतान दंड, फौजदारी शर्तों का विश्लेषण करें।

दस्तावेज़:
{document_text[:10000]}

बिल्कुल इस प्रारूप में उत्तर दें:

🏦 PRE-PAYMENT & FORECLOSURE SHIELD:

PREPAYMENT TERMS DETECTED:
Prepayment Allowed: [हां / नहीं / सशर्त / उल्लिखित नहीं]
Lock-in Period: [अवधि या "उल्लिखित नहीं"]
Prepayment Penalty: [% या एकमुश्त शुल्क या "उल्लिखित नहीं"]
Floating Rate Exemption: [हां — RBI व्यक्तियों पर फ्लोटिंग दर ऋण पर दंड प्रतिबंधित करता है / नहीं / लागू नहीं]

FORECLOSURE TERMS DETECTED:
Foreclosure Allowed: [हां / नहीं / सशर्त]
Foreclosure Charges: [राशि या % या "उल्लिखित नहीं"]
Notice Period Required: [दिन या "उल्लिखित नहीं"]

RBI COMPLIANCE CHECK:
फ्लोटिंग दर ऋण पर पूर्व भुगतान दंड नहीं: [अनुपालन / उल्लंघन / लागू नहीं]
RBI उचित अभ्यास संहिता: [अनुपालन / संभावित उल्लंघन / निर्धारित नहीं]

EXIT STRATEGY RECOMMENDATIONS:
Option 1 — [रणनीति नाम]: [कब उपयोग करें, अपेक्षित लागत, कदम]
Option 2 — [रणनीति नाम]: [कब उपयोग करें, अपेक्षित लागत, कदम]
Option 3 — [रणनीति नाम]: [कब उपयोग करें, अपेक्षित लागत, कदम]

RED FLAGS:
[असामान्य रूप से उच्च निकास बाधाएं बनाने वाली धाराएं — सूची या "कोई नहीं मिला"]{lang_note}"""
    return f"""You are a loan exit strategy advisor with expertise in Indian banking regulations (RBI). Analyze this document for prepayment penalties, foreclosure terms, and compliance with 2024-2026 RBI circulars.

DOCUMENT:
{document_text[:10000]}

Output EXACTLY this format:

🏦 PRE-PAYMENT & FORECLOSURE SHIELD:

PREPAYMENT TERMS DETECTED:
Prepayment Allowed: [Yes / No / Conditional / Not Specified]
Lock-in Period: [duration or "Not specified"]
Prepayment Penalty: [% or flat fee or "Not specified"]
Floating Rate Exemption: [Yes — RBI prohibits penalties on floating rate loans for individuals / No / Not Applicable]

FORECLOSURE TERMS DETECTED:
Foreclosure Allowed: [Yes / No / Conditional]
Foreclosure Charges: [amount or % or "Not specified"]
Notice Period Required: [days or "Not specified"]

RBI COMPLIANCE CHECK (2024-2026 Circulars):
RBI/2023-24/XX — No prepayment penalty on floating rate loans (individuals/MSMEs): [Compliant / Violation / N/A]
RBI Fair Practice Code compliance: [Compliant / Potentially Violated / Cannot Determine]
IRDAI / SEBI applicability: [Yes / No / Not Applicable]

EXIT STRATEGY RECOMMENDATIONS:
Option 1 — [strategy name]: [When to use, expected cost, steps]
Option 2 — [strategy name]: [When to use, expected cost, steps]
Option 3 — [strategy name]: [When to use, expected cost, steps]

RED FLAGS:
[Any clauses that create unusually high exit barriers — list or "None detected"]

If this is not a loan document, output: "Insufficient information to analyze this module — Pre-Payment Shield applies to loan agreements only"
"""


def lc_discrepancy_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप एक व्यापार वित्त विशेषज्ञ और साख पत्र (LC) अनुपालन विशेषज्ञ हैं (UCP 600 / ISBP 745)। LC विसंगतियों का विश्लेषण करें।

दस्तावेज़:
{document_text[:10000]}

बिल्कुल इस प्रारूप में उत्तर दें:

📑 LC DISCREPANCY DETECTOR:

DOCUMENT TYPE IDENTIFIED: [LC / बिल ऑफ लैडिंग / चालान / अन्य / LC दस्तावेज़ नहीं]

DISCREPANCIES FOUND:
[प्रत्येक विसंगति:]
- Discrepancy #[N]: [विवरण] | UCP 600 Reference: [अनुच्छेद संख्या] | Severity: 🔴 भुगतान-अवरोधक / 🟡 माफ करने योग्य / 🟢 मामूली | Fix: [सुधार]

COMMON LC TRAPS CHECK:
देर से प्रस्तुति (शिपमेंट के बाद >21 दिन): [पाया गया / नहीं पाया / निर्धारित नहीं]
बासी बिल ऑफ लैडिंग: [पाया गया / नहीं पाया / निर्धारित नहीं]
चालान और LC के बीच विवरण बेमेल: [पाया गया / नहीं पाया / निर्धारित नहीं]
राशि विसंगति: [पाया गया / नहीं पाया / निर्धारित नहीं]
गलत लोडिंग/डिस्चार्ज बंदरगाह: [पाया गया / नहीं पाया / निर्धारित नहीं]
अनुपस्थित समर्थन: [पाया गया / नहीं पाया / निर्धारित नहीं]

DISCREPANCY VERDICT: [स्वच्छ / मामूली सुधार आवश्यक / भुगतान-अवरोधक समस्याएं मिलीं]{lang_note}"""
    return f"""You are a trade finance specialist and Letter of Credit (LC) compliance expert (UCP 600 / ISBP 745). Analyze this document for LC discrepancies that could block payment.

DOCUMENT:
{document_text[:10000]}

Output EXACTLY this format:

📑 LC DISCREPANCY DETECTOR:

DOCUMENT TYPE IDENTIFIED: [LC / Bill of Lading / Invoice / Other / Not an LC document]

DISCREPANCIES FOUND:
[List each discrepancy:]
- Discrepancy #[N]: [Description] | UCP 600 Reference: [Article number if applicable] | Severity: 🔴 Payment-Blocking / 🟡 Waivable / 🟢 Minor | Fix: [exact correction needed]

COMMON LC TRAPS CHECK:
Late presentation (>21 days after shipment): [Detected / Not Detected / Cannot Determine]
Stale Bill of Lading: [Detected / Not Detected / Cannot Determine]
Description mismatch between invoice and LC: [Detected / Not Detected / Cannot Determine]
Amount discrepancy (+-5% tolerance per UCP 600 Art. 30): [Detected / Not Detected / Cannot Determine]
Wrong port of loading/discharge: [Detected / Not Detected / Cannot Determine]
Missing endorsements: [Detected / Not Detected / Cannot Determine]

PERFECT INVOICE FORMAT SUGGESTION:
[Provide a structured template or checklist for a compliant invoice based on this LC's requirements]

DISCREPANCY VERDICT: [Clean / Minor Fixes Needed / Payment-Blocking Issues Found]

If this is not an LC-related document, output: "Insufficient information to analyze this module — LC Discrepancy Detector applies to Letters of Credit and trade finance documents"
"""


def co_lending_radar_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप RBI के को-लेंडिंग मॉडल (CLM) दिशानिर्देशों में विशेषज्ञता वाले को-लेंडिंग अनुपालन विशेषज्ञ हैं।

दस्तावेज़:
{document_text[:10000]}

बिल्कुल इस प्रारूप में उत्तर दें:

⚖️ CO-LENDING RADAR:

CO-LENDING STRUCTURE DETECTED:
Primary Lender (Bank): [नाम या "उल्लिखित नहीं"]
Co-Lender (NBFC): [नाम या "उल्लिखित नहीं"]
Loan Split Ratio: [जैसे 80:20 बैंक:NBFC या "उल्लिखित नहीं"]
Interest Rate Structure: [स्थिर / फ्लोटिंग / मिश्रित / उल्लिखित नहीं]

RBI CO-LENDING MODEL (CLM) COMPLIANCE:
RBI मास्टर निर्देश — को-लेंडिंग (2020): [अनुपालन / संभावित उल्लंघन / निर्धारित नहीं]
न्यूनतम बैंक हिस्सा (>=80%): [पूरा / पूरा नहीं / लागू नहीं]
उधारकर्ता के लिए एकल ऋण खाता: [पुष्टि / अपुष्टि / उल्लिखित नहीं]

CONFLICTING OBLIGATIONS DETECTED:
[प्रत्येक संघर्ष:]
- Conflict #[N]: [विवरण] | Severity: उच्च / मध्यम / कम | Resolution: [कैसे हल करें]

CO-LENDING RED FLAGS:
[धाराएं जो उधारकर्ता को दोहरी देनदारी के लिए उजागर करती हैं — सूची या "कोई नहीं मिला"]

VERDICT: [अनुपालन संरचना / मामूली समस्याएं / बड़ी संरचनात्मक समस्याएं]{lang_note}"""
    return f"""You are a co-lending compliance specialist with expertise in RBI's Co-Lending Model (CLM) guidelines. Analyze this document for conflicting obligations, unclear splits, and compliance issues between a Bank and NBFC.

DOCUMENT:
{document_text[:10000]}

Output EXACTLY this format:

⚖️ CO-LENDING RADAR:

CO-LENDING STRUCTURE DETECTED:
Primary Lender (Bank): [Name or "Not specified"]
Co-Lender (NBFC): [Name or "Not specified"]
Loan Split Ratio: [e.g., 80:20 Bank:NBFC or "Not specified"]
Interest Rate Structure: [Fixed / Floating / Blended / Not specified]

RBI CO-LENDING MODEL (CLM) COMPLIANCE:
RBI Master Direction — Co-Lending by Banks and NBFCs to Priority Sector (2020): [Compliant / Potentially Violated / Cannot Determine]
Minimum Bank share (>=80% for priority sector): [Met / Not Met / N/A]
Single Loan Account for borrower: [Confirmed / Not Confirmed / Not Specified]
Joint contribution at disbursement: [Confirmed / Not Confirmed / Not Specified]

CONFLICTING OBLIGATIONS DETECTED:
[List each conflict:]
- Conflict #[N]: [Description] | Severity: High / Medium / Low | Resolution Suggestion: [how to resolve]

POINTS OF CONFUSION FOR BORROWER:
[List any clauses that may confuse the borrower about who to pay, who to contact, or who is responsible]

RISK ALLOCATION ANALYSIS:
Credit Risk: [Who bears it and in what proportion]
Operational Risk: [Who handles servicing, collections]
Regulatory Risk: [Who is liable if RBI action is taken]

CO-LENDING RED FLAGS:
[Any clauses that expose the borrower to double liability, unclear recourse, or regulatory violations — or "None detected"]

VERDICT: [Compliant Structure / Minor Issues / Major Structural Problems]

If this is not a co-lending document, output: "Insufficient information to analyze this module — Co-Lending Radar applies to co-lending and joint lending agreements only"
"""


# ─────────────────────────────────────────────────────────────
# FORM FILLER AGENT PROMPT
# ─────────────────────────────────────────────────────────────

def formfill_prompt(document_text: str, lang: str = "en") -> str:
    lang_note = _lang_instruction(lang)
    if lang == "hi":
        return f"""आप एक सटीक दस्तावेज़ विश्लेषण सहायक हैं।
आपका कार्य: केवल उन फ़ॉर्म फ़ील्ड की पहचान करें जो दस्तावेज़ में स्पष्ट रूप से मौजूद हैं और एक "इस फ़ॉर्म को कैसे भरें" मार्गदर्शिका बनाएं।

कड़े नियम (अनिवार्य):
1. कोई भी फ़ील्ड न बनाएं, न मानें, न अनुमान लगाएं।
2. केवल वही फ़ील्ड शामिल करें जो दस्तावेज़ में स्पष्ट रूप से मौजूद हो।
3. हर फ़ील्ड के लिए दस्तावेज़ से सटीक उद्धरण दें।
4. यदि प्रत्यक्ष प्रमाण नहीं मिला, तो वह फ़ील्ड शामिल न करें।
5. सामान्य फ़ॉर्म ज्ञान का उपयोग न करें।
6. यदि कोई फ़ील्ड नहीं मिली: "इस दस्तावेज़ में कोई फ़ॉर्म फ़ील्ड नहीं मिली।" लिखें।
7. यदि कोई जानकारी अस्पष्ट हो: "अपर्याप्त जानकारी" लिखें।

आउटपुट प्रारूप (सख्त):

🧾 इस फ़ॉर्म को कैसे भरें:

1. फ़ील्ड का नाम:
   - दस्तावेज़ में मिला: "<दस्तावेज़ से सटीक उद्धरण>"
   - क्या भरें:
   - प्रारूप:
   - उदाहरण:
   - सुझाव:

2. फ़ील्ड का नाम:
   [समान प्रारूप]

📋 त्वरित सारांश:
[फ़ॉर्म में कुल फ़ील्ड की संख्या और सबसे महत्वपूर्ण जानकारी 2-3 वाक्यों में]

⚠️ महत्वपूर्ण नोट:
[कोई विशेष निर्देश, समय सीमा, या अनिवार्य फ़ील्ड यदि दस्तावेज़ में उल्लिखित हो]

दस्तावेज़:
----------------
{document_text[:12000]}
----------------
{lang_note}"""

    return f"""You are a precise document analysis assistant.
Your task is to extract ONLY form fields that are explicitly present in the given document and generate a "How to Fill This Form" guide.

STRICT RULES (MANDATORY):
1. DO NOT invent, assume, or infer any fields.
2. ONLY include a field if it is clearly present in the document text.
3. For EVERY field, you MUST provide an exact quote from the document as evidence.
4. If you cannot find direct evidence, DO NOT include that field.
5. DO NOT use general knowledge about common forms.
6. If no fields are clearly present, respond exactly with: "No form fields detected in this document."
7. Keep explanations simple and factual.
8. If any information is unclear, say: "Insufficient information"

OUTPUT FORMAT (STRICT):

🧾 How to Fill This Form:

1. Field Name:
   - Found in text: "<exact quote from document>"
   - What to enter:
   - Format:
   - Example:
   - Tip:

2. Field Name:
   [same format]

📋 Quick Summary:
[Total number of fields found and most critical info in 2-3 sentences]

⚠️ Important Notes:
[Any special instructions, deadlines, or mandatory fields if explicitly mentioned in the document]

FINAL CHECK (VERY IMPORTANT):
Before giving the answer, verify:
- Every field has a supporting quote
- No extra or assumed fields are included
If any field does not meet these conditions, REMOVE it.

Document:
----------------
{document_text[:12000]}
----------------
{lang_note}"""