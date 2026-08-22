import os
import re
import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import faiss

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Kreative Kudi AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "chunks.csv"
LOGO_FILE = BASE_DIR / "face c2.png"


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if api_key:
    try:
        gemini_client = genai.Client(api_key=api_key)
    except Exception:
        gemini_client = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .block-container {
        max-width: 900px;
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
    }

    header {
        display: none !important;
    }

    /* Hide the streamlit header/toolbar */
    [data-testid="stHeader"] {
        display: none !important;
    }


    /* ========================================================
       BRAND HEADER
       ======================================================== */

    .kk-brand-header {
        position: sticky;
        top: 0;
        z-index: 999;
        overflow: hidden;
        padding: 20px 0;
        margin-bottom: 20px;
        background: linear-gradient(110deg, rgba(108, 45, 210, 0.13), rgba(224, 47, 153, 0.10), rgba(255, 142, 30, 0.08), rgba(0, 188, 190, 0.12));
        border-bottom: 1px solid rgba(120, 70, 220, 0.10);
    }

    .kk-brand-header::after {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        height: 6px;
        background: linear-gradient(90deg, rgba(108, 45, 210, 0.70), rgba(224, 47, 153, 0.60), rgba(255, 142, 30, 0.55), rgba(0, 188, 190, 0.65));
        opacity: 0.75;
    }

    .kk-brand-inner {
        max-width: 900px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 0 20px;
    }

    .kk-brand-logo {
        width: 65px;
        height: 65px;
        object-fit: contain;
        flex-shrink: 0;
    }

    .kk-brand-text {
        flex: 1;
    }

    .kk-brand-title {
        font-size: 32px;
        font-weight: 800;
        line-height: 1.2;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .kk-brand-title-gradient {
        background: linear-gradient(90deg, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .kk-brand-title-black {
        color: #1a1a1a;
    }

    .kk-brand-welcome {
        margin-top: 6px;
        font-size: 14px;
        color: #777b87;
        font-weight: 500;
    }


    /* ========================================================
       WELCOME CARD
       ======================================================== */

    .kk-welcome {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid rgba(110, 70, 220, 0.12);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(70, 40, 130, 0.08);
        display: flex;
        gap: 20px;
        align-items: flex-start;
    }

    .kk-welcome-icon {
        font-size: 48px;
        flex-shrink: 0;
        width: 70px;
        height: 70px;
        background: rgba(108, 45, 210, 0.08);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .kk-welcome-content {
        flex: 1;
    }

    .kk-welcome-title {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 8px;
        letter-spacing: -0.3px;
    }

    .kk-welcome-text {
        font-size: 14px;
        color: #777b87;
        line-height: 1.6;
    }


    /* ========================================================
       QUICK QUESTIONS
       ======================================================== */

    .kk-section-title {
        font-size: 15px;
        font-weight: 700;
        color: #7c3aed;
        margin: 0 0 16px 0;
        display: flex;
        align-items: center;
        gap: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Streamlit columns container */
    div[data-testid="column"] {
        display: block;
    }


    /* Base quick question card */

    .kk-question-card {
        display: flex;
        gap: 16px;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid transparent;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        background: #f5f5f5 !important;
        min-height: 70px;
        align-items: flex-start;
        position: relative;
        z-index: 1;
        pointer-events: none;
    }

    .kk-question-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }

    .kk-question-icon {
        font-size: 32px;
        width: 48px;
        height: 48px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
    }

    .kk-question-content {
        flex: 1;
        min-width: 0;
    }

    .kk-question-title {
        font-size: 15px;
        font-weight: 700;
        margin: 0 0 4px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .kk-question-description {
        font-size: 13px;
        line-height: 1.4;
        margin: 0;
        color: #555;
    }

    .kk-question-arrow {
        font-size: 16px;
        opacity: 0.6;
        transition: opacity 0.2s ease;
    }

    .kk-question-card:hover .kk-question-arrow {
        opacity: 1;
    }


    /* Founder Card */
    .kk-question-founder {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.10), rgba(168, 85, 247, 0.05)) !important;
        border-color: rgba(124, 58, 237, 0.20) !important;
    }

    .kk-question-founder .kk-question-icon {
        background: rgba(124, 58, 237, 0.15);
        color: #7c3aed;
    }

    .kk-question-founder .kk-question-title {
        color: #7c3aed;
    }


    /* Location Card */
    .kk-question-location {
        background: linear-gradient(135deg, rgba(219, 39, 119, 0.10), rgba(244, 114, 182, 0.05)) !important;
        border-color: rgba(219, 39, 119, 0.20) !important;
    }

    .kk-question-location .kk-question-icon {
        background: rgba(219, 39, 119, 0.15);
        color: #db2777;
    }

    .kk-question-location .kk-question-title {
        color: #db2777;
    }


    /* Services Card */
    .kk-question-services {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.10), rgba(251, 146, 60, 0.05)) !important;
        border-color: rgba(249, 115, 22, 0.20) !important;
    }

    .kk-question-services .kk-question-icon {
        background: rgba(249, 115, 22, 0.15);
        color: #f97316;
    }

    .kk-question-services .kk-question-title {
        color: #f97316;
    }


    /* Courses Card */
    .kk-question-courses {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.10), rgba(52, 211, 153, 0.05)) !important;
        border-color: rgba(16, 185, 129, 0.20) !important;
    }

    .kk-question-courses .kk-question-icon {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
    }

    .kk-question-courses .kk-question-title {
        color: #10b981;
    }


    /* ========================================================
       POWERED BY
       ======================================================== */

    .kk-powered {
        text-align: center;
        margin-top: 16px;
        margin-bottom: 12px;
        padding-top: 16px;
        padding-bottom: 12px;
        border-top: 1px solid rgba(120, 120, 140, 0.15);
        color: #888b95;
        font-size: 12px;
        font-weight: 500;
    }

    .kk-powered strong {
        color: #7c3aed;
        font-weight: 600;
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {
        margin-top: 16px;
        margin-bottom: 8px;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 20px !important;
        border: 1px solid rgba(124, 58, 237, 0.18) !important;
        background: #ffffff !important;
        box-shadow: 0 4px 16px rgba(70, 40, 130, 0.08) !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
    }

    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #7c3aed, #db2777) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        min-width: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    [data-testid="stChatMessage"] {
        padding-top: 8px;
        padding-bottom: 8px;
    }


    /* ========================================================
       SOURCE CARDS
       ======================================================== */

    .source-card {
        border: 1px solid #eeeeee;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
        background: #fafafa;
    }

    .source-title {
        font-size: 13px;
        font-weight: 650;
        color: #1a1a1a;
    }

    .source-score {
        font-size: 11px;
        color: #888;
        margin-top: 4px;
    }


    /* ========================================================
       HIDE STREAMLIT BUTTONS AND USE CUSTOM STYLING
       ======================================================== */

    div[data-testid="stButton"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stButton"] button {
        width: 100% !important;
        height: auto !important;
        min-height: 70px !important;
        padding: 16px 18px !important;
        border-radius: 16px !important;
        border: 2px solid rgba(124, 58, 237, 0.25) !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-align: left !important;
        white-space: pre-wrap !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 6px !important;
        cursor: pointer !important;
        background: linear-gradient(135deg, #ede9fe 0%, #f5f3ff 100%) !important;
        color: #7c3aed !important;
    }

    div[data-testid="stButton"] button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
    }

    /* Founder Button - Light Purple */
    div[data-testid="stButton"]:nth-of-type(1) button,
    div[data-testid="stButton"] button:nth-of-type(1) {
        background: linear-gradient(135deg, #ede9fe 0%, #f5f3ff 100%) !important;
        border: 2px solid rgba(124, 58, 237, 0.25) !important;
        color: #7c3aed !important;
    }

    /* Location Button - Light Pink */
    div[data-testid="stButton"]:nth-of-type(2) button,
    div[data-testid="stButton"] button:nth-of-type(2) {
        background: linear-gradient(135deg, #fce7f3 0%, #fbf1f7 100%) !important;
        border: 2px solid rgba(219, 39, 119, 0.25) !important;
        color: #db2777 !important;
    }

    /* Services Button - Light Orange/Peach */
    div[data-testid="stButton"]:nth-of-type(3) button,
    div[data-testid="stButton"] button:nth-of-type(3) {
        background: linear-gradient(135deg, #fed7aa 0%, #fef3c7 100%) !important;
        border: 2px solid rgba(249, 115, 22, 0.25) !important;
        color: #f97316 !important;
    }

    /* Courses Button - Light Teal/Cyan */
    div[data-testid="stButton"]:nth-of-type(4) button,
    div[data-testid="stButton"] button:nth-of-type(4) {
        background: linear-gradient(135deg, #ccfbf1 0%, #d1fae5 100%) !important;
        border: 2px solid rgba(16, 185, 129, 0.25) !important;
        color: #10b981 !important;
    }


    @media (max-width: 768px) {

        .block-container {
            padding-left: 12px !important;
            padding-right: 12px !important;
            padding-top: 0.5rem !important;
        }

        .kk-brand-header {
            padding: 12px 12px;
            margin-bottom: 12px;
        }

        .kk-brand-inner {
            gap: 10px;
        }

        .kk-brand-logo {
            width: 50px;
            height: 50px;
        }

        .kk-brand-title {
            font-size: 24px;
        }

        .kk-brand-welcome {
            font-size: 12px;
            margin-top: 3px;
        }

        .kk-welcome {
            padding: 14px;
            border-radius: 14px;
            margin-bottom: 14px;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }

        .kk-welcome-icon {
            width: 52px;
            height: 52px;
            font-size: 32px;
        }

        .kk-welcome-title {
            font-size: 16px;
        }

        .kk-welcome-text {
            font-size: 12px;
        }

        .kk-section-title {
            font-size: 12px;
            margin-bottom: 10px;
        }

        .kk-question-card {
            padding: 12px;
            gap: 12px;
            min-height: 60px;
            font-size: 14px;
        }

        .kk-question-icon {
            font-size: 26px;
            width: 38px;
            height: 38px;
        }

        .kk-question-title {
            font-size: 13px;
        }

        .kk-question-description {
            font-size: 11px;
        }

        .kk-powered {
            font-size: 9px;
            margin-top: 10px;
            padding-top: 10px;
        }

    }

    /* ========================================================
       PREVENT AUTO-SCROLL
       ======================================================== */

    html, body {
        scroll-behavior: auto !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model


embedding_model = load_embedding_model()


# ============================================================
# LOAD CHUNKS
# ============================================================

@st.cache_data
def load_chunks():
    if not CHUNKS_FILE.exists():
        st.error(f"chunks.csv was not found at:\n{CHUNKS_FILE}")
        st.stop()

    df = pd.read_csv(CHUNKS_FILE)
    return df


chunks_df = load_chunks()


# ============================================================
# CREATE FAISS INDEX
# ============================================================

@st.cache_resource
def create_faiss_index(_chunks_df, _embedding_model):
    texts = (_chunks_df["text"].fillna("").astype(str).tolist())
    embeddings = _embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


index = create_faiss_index(chunks_df, embedding_model)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_gemini_error" not in st.session_state:
    st.session_state.last_gemini_error = None


# ============================================================
# CONVERSATION CONTEXT
# ============================================================

def get_conversation_context():
    return st.session_state.messages[-6:]


def build_search_query(query):
    query = query.strip()

    if not st.session_state.messages:
        return query

    history = get_conversation_context()
    previous_user_messages = [msg["content"] for msg in history if msg["role"] == "user"]

    if not previous_user_messages:
        return query

    last_user_question = previous_user_messages[-1]

    followup_patterns = [
        r"\bit\b", r"\bthis\b", r"\bthat\b", r"\bthey\b", r"\bthem\b", r"\btheir\b",
        r"\bhow much\b", r"\bhow much does\b",
        r"\bwhat is the price\b", r"\bprice\b", r"\bcost\b", r"\bcharges\b", r"\brate\b", r"\bfee\b", r"\bfees\b",
        r"\bhow long\b", r"\bmore details\b", r"\bmore information\b", r"\bwhat about\b", r"\band what about\b"
    ]

    is_followup = any(re.search(pattern, query.lower()) for pattern in followup_patterns)

    if is_followup:
        return last_user_question + " " + query

    return query


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(question, top_k=5):
    search_query = build_search_query(question)
    question_embedding = embedding_model.encode([search_query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    search_k = min(10, len(chunks_df))
    scores, indices = index.search(question_embedding, search_k)

    results = []
    question_lower = search_query.lower()
    question_words = set(word.strip(".,?!:;()[]{}\"'") for word in question_lower.split() if len(word) > 2)

    for semantic_score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = chunks_df.iloc[idx]
        text = str(row.get("text", ""))
        text_lower = text.lower()

        text_words = set(word.strip(".,?!:;()[]{}\"'") for word in text_lower.split())

        if question_words:
            keyword_score = len(question_words.intersection(text_words)) / len(question_words)
        else:
            keyword_score = 0.0

        page_type = str(row.get("page_type", "")).lower()
        title = str(row.get("title", "")).lower()
        boost = 0.0

        if any(word in question_lower for word in ["founder", "owner", "ceo", "who started"]):
            if page_type == "founder":
                boost += 0.25
            if "founder" in title or "ceo" in title:
                boost += 0.10

        if any(word in question_lower for word in ["established", "started", "founded", "since", "when"]):
            if page_type in ["company_services", "about"]:
                boost += 0.20

        if any(word in question_lower for word in ["course", "courses", "training", "learn", "institute"]):
            if page_type == "training":
                boost += 0.25

        if any(word in question_lower for word in ["contact", "email", "phone", "address", "location", "reach", "city", "based"]):
            if page_type == "contact":
                boost += 0.25

        if any(word in question_lower for word in ["price", "pricing", "cost", "fee", "fees", "charge", "charges", "rate"]):
            if any(word in text_lower for word in ["price", "pricing", "cost", "fee", "fees", "charge", "charges", "rate", "2999", "₹", "rs"]):
                boost += 0.15

        final_score = 0.65 * float(semantic_score) + 0.20 * float(keyword_score) + boost

        results.append({
            "chunk_id": row.get("chunk_id", ""),
            "document_id": row.get("document_id", ""),
            "title": row.get("title", ""),
            "source_url": row.get("source_url", ""),
            "page_type": row.get("page_type", ""),
            "text": text,
            "semantic_score": float(semantic_score),
            "keyword_score": float(keyword_score),
            "boost": float(boost),
            "final_score": float(final_score)
        })

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]


# ============================================================
# BUILD GEMINI CONTEXT
# ============================================================

def build_context(results):
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(f"""
SOURCE {i}

Document:
{result['document_id']}

Title:
{result['title']}

URL:
{result['source_url']}

Content:
{result['text']}
""")
    return "\n\n".join(context_parts)


# ============================================================
# BUILD CONVERSATION HISTORY
# ============================================================

def build_conversation_history():
    history = get_conversation_context()
    if not history:
        return "No previous conversation."
    parts = []
    for message in history:
        role = message["role"]
        content = message["content"]
        if role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)


# ============================================================
# LOCAL RAG FALLBACK
# ============================================================

def local_rag_answer(query, results):
    if not results:
        return "I couldn't find that information in the Kreative Kudi knowledge base."

    search_query = build_search_query(query)
    query_words = [word.lower().strip(".,?!:;()[]{}\"'") for word in search_query.split() if len(word) > 2]
    
    stop_words = {"what", "where", "when", "who", "how", "does", "can", "could", "would", "please", "tell", "about", "the", "and", "for", "with", "this", "that", "are", "is", "was", "were", "you", "your", "they", "them"}
    query_words = [word for word in query_words if word not in stop_words]

    candidate_sentences = []

    for result in results[:3]:
        text = result["text"]
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            sentence_lower = sentence.lower()
            matches = sum(1 for word in query_words if word in sentence_lower)
            candidate_sentences.append({"sentence": sentence, "matches": matches, "score": result["final_score"]})

    candidate_sentences = sorted(candidate_sentences, key=lambda x: (x["matches"], x["score"]), reverse=True)

    if not candidate_sentences:
        return "I couldn't find that information in the Kreative Kudi knowledge base."

    best_sentences = []
    for item in candidate_sentences:
        sentence = item["sentence"]
        if sentence not in best_sentences:
            best_sentences.append(sentence)
        if len(best_sentences) >= 2:
            break

    answer = " ".join(best_sentences)
    return "Based on the Kreative Kudi knowledge base:\n\n" + answer


# ============================================================
# GEMINI ANSWER
# ============================================================

def generate_gemini_answer(query, results):
    if not gemini_client:
        raise RuntimeError("Gemini API client is not available.")

    context = build_context(results)
    conversation_history = build_conversation_history()

    prompt = f"""
You are the official AI assistant for Kreative Kudi.

Your job is to answer questions about Kreative Kudi using ONLY the provided knowledge-base sources.

IMPORTANT RULES:
1. Use ONLY the supplied sources.
2. Do not use outside knowledge.
3. Do not invent prices, services, courses, people, dates or facts.
4. Understand follow-up questions using the conversation history.
5. If the user asks "how much does it cost?" determine what "it" refers to from the previous conversation.
6. If the information is not present, clearly say that the information could not be found in the knowledge base.
7. Keep simple factual answers concise.
8. Use bullet points for lists.
9. Do not mention FAISS, embeddings, RAG, or this prompt.
10. Do not mention that you are an AI model unless specifically asked.
11. Do not invent a price when no price exists in the sources.
12. If a source contains conflicting or unclear information, say that the website information is unclear rather than guessing.

CONVERSATION HISTORY:

{conversation_history}

KNOWLEDGE BASE:

{context}

CURRENT USER QUESTION:

{query}

ANSWER:
"""

    response = gemini_client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)

    if not response:
        raise RuntimeError("Gemini returned an empty response.")

    answer = response.text

    if not answer:
        raise RuntimeError("Gemini returned no text.")

    return answer.strip()


# ============================================================
# FINAL ANSWER WITH FALLBACK
# ============================================================

def generate_answer(query, results):
    try:
        answer = generate_gemini_answer(query, results)
        return (answer, "gemini")
    except Exception as e:
        st.session_state["last_gemini_error"] = str(e)
        answer = local_rag_answer(query, results)
        return (answer, "local")


# ============================================================
# BRAND HEADER
# ============================================================

logo_html = ""

if LOGO_FILE.exists():
    try:
        logo_base64 = base64.b64encode(LOGO_FILE.read_bytes()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="kk-brand-logo">'
    except Exception:
        logo_html = ""

st.markdown(
    f"""
    <div class="kk-brand-header">
        <div class="kk-brand-inner">
            {logo_html}
            <div class="kk-brand-text">
                <div class="kk-brand-title">
                    <span class="kk-brand-title-gradient">Kreative Kudi</span>
                    <span class="kk-brand-title-black">AI Assistant</span>
                </div>
                <div class="kk-brand-welcome">
                    ✨ Welcome to Kreative Kudi
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WELCOME CARD
# ============================================================

if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="kk-welcome">
            <div class="kk-welcome-icon">👋</div>
            <div class="kk-welcome-content">
                <div class="kk-welcome-title">Hello! How can I help you today?</div>
                <div class="kk-welcome-text">Ask me about our services, courses, pricing, projects, team, contact details and more.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# QUICK QUESTIONS
# ============================================================

if len(st.session_state.messages) == 0:
    st.markdown('<div class="kk-section-title">✨ Quick questions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="small")

    with col1:
        if st.button("👤 Founder\nWho is the founder of Kreative Kudi?", key="q_founder", use_container_width=True):
            st.session_state["suggested_query"] = "Who is the founder of Kreative Kudi?"
            st.rerun()

    with col2:
        if st.button("📍 Location\nWhere is Kreative Kudi based?", key="q_location", use_container_width=True):
            st.session_state["suggested_query"] = "Where is Kreative Kudi based?"
            st.rerun()

    col3, col4 = st.columns(2, gap="small")

    with col3:
        if st.button("🎨 Services\nWhat services does Kreative Kudi offer?", key="q_services", use_container_width=True):
            st.session_state["suggested_query"] = "What services does Kreative Kudi offer?"
            st.rerun()

    with col4:
        if st.button("📚 Courses\nWhat courses are available?", key="q_courses", use_container_width=True):
            st.session_state["suggested_query"] = "What courses are available at Kreative Kudi?"
            st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    title = source.get("title", "Kreative Kudi")
                    url = source.get("source_url", "")
                    score = source.get("final_score", source.get("semantic_score", 0))

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-title">{title}</div>
                            <div class="source-score">Relevance: {score:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if url:
                        st.markdown(f"[🔗 View source]({url})")

        if message["role"] == "assistant" and message.get("generation_mode") == "local":
            st.caption("⚡ Gemini was temporarily unavailable. Answer generated from the local Kreative Kudi knowledge base.")


# ============================================================
# USER INPUT
# ============================================================

suggested_query = st.session_state.pop("suggested_query", None)
query = st.chat_input("Ask something about Kreative Kudi...")

# Add powered by at the bottom
st.markdown(
    '<div class="kk-powered">🛡️ Powered by <strong>Kreative Kudi Knowledge Base</strong></div>',
    unsafe_allow_html=True
)

if suggested_query:
    query = suggested_query


# ============================================================
# PROCESS QUESTION
# ============================================================

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching Kreative Kudi knowledge base..."):
            results = retrieve_documents(query, top_k=5)

        with st.spinner("Preparing your answer..."):
            answer, generation_mode = generate_answer(query, results)

        st.markdown(answer)

        if generation_mode == "local":
            st.info("Gemini is temporarily unavailable, so I answered using the local Kreative Kudi knowledge base.")

        if results:
            with st.expander("📚 Sources"):
                for source in results[:3]:
                    title = source.get("title", "Kreative Kudi")
                    url = source.get("source_url", "")
                    score = source.get("final_score", 0)

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-title">{title}</div>
                            <div class="source-score">Relevance: {score:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if url:
                        st.markdown(f"[🔗 View source]({url})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": results[:3],
        "generation_mode": generation_mode
    })

    st.rerun()