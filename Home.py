"""Neo Smart Living — Synthetic Market Research Dashboard (Home / Landing Page)."""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Neo Smart Living — Research Dashboard",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Data paths
PROTO_OUTPUT = Path(__file__).parent.parent / "prototype" / "output"
SURVEY_DATA = PROTO_OUTPUT / "synthetic_responses.csv"
INTERVIEW_DATA = PROTO_OUTPUT / "interview_analysis.csv"
TRANSCRIPTS_DATA = PROTO_OUTPUT / "interview_transcripts.csv"

# --- Hero ---
st.title("🏡 Neo Smart Living")
st.subheader("Tahoe Mini — Synthetic Market Research Platform")
st.markdown(
    """
    Traditional market research for the **Tahoe Mini** — a 117 sq ft prefab backyard structure at **$23,000** —
    would cost **$6,000–$48,000** in surveys and interviews.
    This platform replaces that fixed cost with **~$2–$4 in API calls**, generating equivalent synthetic customer
    data and letting you ask questions about your customers on demand.
    """
)

st.divider()

# --- Key Stats ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Interviews", "30", help="Synthetic depth interviews via GPT-4.1-mini & Gemini 2.5-Flash")
col2.metric("Survey Respondents", "60", help="6 per segment × 5 segments × 2 models")
col3.metric("Market Segments", "5", help="Remote Professional, Active Adventurer, Wellness Seeker, Property Maximizer, Budget DIYer")
col4.metric("LLM Models", "2", help="OpenAI GPT-4.1-mini and Google Gemini 2.5-Flash via OpenRouter")
col5.metric("Product Price", "$23K", help="Tahoe Mini — delivered, installed, warranted")

st.divider()

# --- Cost & Methodology ---
cost_col, stamp_col = st.columns([1, 1])

with cost_col:
    st.info(
        "**Traditional vs. synthetic research cost**\n\n"
        "| | Traditional | This Pipeline |\n"
        "|---|---|---|\n"
        "| 30 depth interviews | $3,000–$9,000 | ~$0.50 |\n"
        "| 60 survey responses | $12,000–$24,000 | ~$1.50 |\n"
        "| Analysis & reporting | $2,000–$10,000 | ~$0.50 |\n"
        "| **Total** | **$6K–$48K+** | **~$2–$4** |\n\n"
        "Instead of paying a steep fixed cost for real surveys, you pay only for API calls — "
        "then validate only the highest-signal findings with real respondents.\n\n"
        "_Synthetic findings are directional hypotheses, not decision-grade evidence._"
    )

with stamp_col:
    st.success(
        "**STAMP Dual-LLM Validation**\n\n"
        "Every question is answered independently by **GPT-4.1-mini** and **Gemini 2.5 Flash**. "
        "A **Claude Sonnet** judge then compares the two answers and classifies any disagreement:\n\n"
        "- **Models agree** → High-confidence finding\n"
        "- **Minor disagreement** → Emphasis or framing difference\n"
        "- **Major disagreement** → Potential hallucination or factual conflict — needs validation\n\n"
        "_Based on Lin (under review) — Structured Taxonomy AI Measurement Protocol._"
    )

st.divider()

# --- Pipeline Flow ---
st.subheader("How the Platform Works")
p1, p2, p3, p4 = st.columns(4)

with p1:
    st.markdown(
        "**1 → Qualitative**\n\n"
        "30 synthetic depth interviews with SoCal homeowner personas probe backyard needs, "
        "unmet wants, and emotional reactions to the Tahoe Mini. "
        "LDA topic modeling + emotion classification surface emergent themes and segments."
    )

with p2:
    st.markdown(
        "**2 → Quantitative**\n\n"
        "Interview-derived segments inform 5 psychographic profiles. "
        "60 synthetic respondents complete a 35-item survey. "
        "Mann-Whitney U tests flag variables where GPT and Gemini diverge."
    )

with p3:
    st.markdown(
        "**3 → Building Codes**\n\n"
        "A RAG chatbot answers permit and zoning questions using uploaded building code documents. "
        "GPT-4.1-mini and Gemini answer independently; Claude Sonnet synthesizes and classifies "
        "disagreements using STAMP."
    )

with p4:
    st.markdown(
        "**4 → Customer Intelligence**\n\n"
        "Ask any question about your customers. Two LLMs retrieve and analyze the synthetic data "
        "independently; a judge synthesizes the answer and scores inter-rater reliability "
        "(Krippendorff's α) so you know how much to trust the finding."
    )

st.divider()

# --- Navigation Cards ---
st.subheader("Explore the Research")

card1, card2, card3, card4 = st.columns(4)

with card1:
    st.markdown("### 🎤 Qualitative Interview Insights")
    st.markdown(
        """
        Explore 30 synthetic depth interviews with SoCal homeowners.

        - **Sentiment analysis** — VADER scores per question
        - **Emotional tone** — excitement, skepticism, curiosity, and more
        - **Thematic analysis** — LDA topics + LLM-extracted themes
        - **Segment discovery** — emergent clusters from interview responses
        - **Full transcripts** — searchable and filterable
        """
    )
    st.page_link("pages/1_Interviews.py", label="Open Interview Insights →", icon="🎤")

with card2:
    st.markdown("### 📊 Quantitative Survey Analytics")
    st.markdown(
        """
        Analyze 60 synthetic survey responses across 5 market segments.

        - **Purchase interest** — Q1/Q2 by segment, income, and work arrangement
        - **Use cases & barriers** — severity heatmap, permit-light positioning impact
        - **Concept test** — 5 positioning concepts rated by appeal and likelihood
        - **Value drivers** — permit-light, install speed, build quality
        - **Model comparison** — GPT vs. Gemini with Mann-Whitney U statistics
        """
    )
    st.page_link("pages/2_Survey.py", label="Open Survey Analytics →", icon="📊")

with card3:
    st.markdown("### 🏗️ Building Codes Assistant")
    st.markdown(
        """
        Ask questions about building codes and permit requirements.

        - **RAG-powered** — answers grounded in uploaded building code documents
        - **Dual-LLM validation** — GPT-4.1-mini and Gemini 2.5 Flash answer independently
        - **Judge synthesis** — Claude Sonnet compares, synthesizes, and flags disagreements
        - **Evaluation mode** — batch accuracy scoring against ground truth Q&A
        """
    )
    st.page_link("pages/3_Building_Codes.py", label="Open Building Codes Assistant →", icon="🏗️")

with card4:
    st.markdown("### 💡 Customer Intelligence Chat")
    st.markdown(
        """
        Ask anything about your customers — get STAMP-validated answers.

        - **Ask your data** — freeform questions over synthetic interviews + surveys
        - **Dual-LLM answers** — GPT-4.1-mini and Gemini answer independently
        - **Judge synthesis** — Claude Sonnet synthesizes and classifies disagreements
        - **Reliability score** — Krippendorff's α tells you how much to trust the finding
        """
    )
    st.page_link("pages/4_Customer_Insights.py", label="Open Customer Intelligence →", icon="💡")

st.divider()

# --- Data Status ---
st.subheader("Data Status")

status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)

with status_col1:
    if SURVEY_DATA.exists():
        st.success(f"✅ Survey data found  \n`{SURVEY_DATA.name}`")
    else:
        st.error(f"❌ Survey data missing  \nRun `python prototype/synthetic_respondents.py`  \nor `python prototype/generate_test_data.py`")

with status_col2:
    if INTERVIEW_DATA.exists():
        st.success(f"✅ Interview analysis found  \n`{INTERVIEW_DATA.name}`")
    elif TRANSCRIPTS_DATA.exists():
        st.warning(f"⚠️ Only raw transcripts found  \nRun `python prototype/interview_analysis.py` for full insights")
    else:
        st.error(f"❌ Interview data missing  \nRun `python prototype/generate_test_interviews.py`")

with status_col3:
    themes_path = PROTO_OUTPUT / "interview_themes.json"
    if themes_path.exists():
        st.success(f"✅ Themes data found  \n`{themes_path.name}`")
    else:
        st.warning(f"⚠️ No themes data  \nRun `python prototype/interview_analysis.py` for thematic analysis")

with status_col4:
    bc_dir = Path(__file__).parent / "data" / "building_codes"
    bc_docs = (
        list(bc_dir.glob("*.pdf")) + list(bc_dir.glob("*.txt")) + list(bc_dir.glob("*.md"))
        if bc_dir.exists() else []
    )
    if bc_docs:
        st.success(f"✅ {len(bc_docs)} building code doc(s)  \nReady for RAG chatbot")
    else:
        st.warning(f"⚠️ No building codes docs  \nAdd PDFs/text to `data/building_codes/`")

with status_col5:
    customer_ready = SURVEY_DATA.exists() and INTERVIEW_DATA.exists()
    if customer_ready:
        st.success("✅ Synthetic data ready  \nCustomer Intelligence Chat enabled")
    else:
        st.warning("⚠️ Synthetic data missing  \nRun prototype scripts to generate data")

st.divider()

# --- About ---
with st.expander("About this project"):
    st.markdown(
        """
        **Neo Smart Living × Aytm Joint Challenge — CPP AI Hackathon 2026**

        Instead of conducting expensive traditional market research ($6K–$48K), this pipeline generates
        realistic synthetic respondent data for ~$2–$4 using AI, enabling rapid testing and iteration.

        **Product:** Tahoe Mini backyard prefab unit
        - Size: 117 sq ft (9 ft × 13 ft)
        - Price: $23,000 (delivered, installed, warranted)
        - Installation: One day, professional
        - Permit: Permit-light (often no building permit required at 117 sq ft in California)

        **Market Segments:**
        1. 🖥️ **Remote Professional** — WFH knowledge workers needing a dedicated home office
        2. 🏔️ **Active Adventurer** — Outdoor enthusiasts needing gear storage and a basecamp
        3. 🧘 **Wellness Seeker** — Fitness/meditation practitioners wanting a personal retreat
        4. 💰 **Property Maximizer** — Higher-income investors focused on ROI and rental income
        5. 🔧 **Budget-Conscious DIYer** — Practical, cost-focused, handy homeowners
        """
    )
