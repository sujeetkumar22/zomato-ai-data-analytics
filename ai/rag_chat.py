"""
Zomato Review RAG Chat Assistant
================================
Interactive Streamlit application providing Retrieval-Augmented Generation (RAG)
over customer reviews from Snowflake (ZOMATO.STAGING.STG_REVIEWS).

Features:
- Live Snowflake data retrieval with Parquet disk caching for instant subsequent launches.
- Vector semantic search using OpenAI text-embedding-3-small (with automatic TF-IDF fallback
  if OpenAI quota is exhausted or offline).
- Conversational chat interface (st.chat_message) with session history.
- Contextual answer synthesis using OpenAI gpt-4o-mini (with intelligent analytical fallback).
- Interactive sidebar with dataset analytics, city filters, and sample question prompts.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Zomato AI Review Intelligence",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Environment & Setup
# ---------------------------------------------------------
_curr = Path(__file__).resolve().parent
load_dotenv(_curr / ".env")
load_dotenv(_curr.parent / "airflow" / ".env")
load_dotenv(_curr.parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_chat")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_SAMPLE_SIZE = 500
TOP_K = 5
CACHE_FILE = _curr / "review_embeddings.parquet"
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# ---------------------------------------------------------
# Snowflake Data Ingestion
# ---------------------------------------------------------
def get_snowflake_connection():
    """Establish Snowflake connection using environment variables."""
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    database = os.getenv("SNOWFLAKE_DATABASE", "ZOMATO")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "ZOMATO_WH")
    role = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")

    return snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        database=database,
        warehouse=warehouse,
        role=role,
    )


def read_reviews_from_snowflake(limit: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    """Fetch reviews from ZOMATO.STAGING.STG_REVIEWS."""
    conn = get_snowflake_connection()
    try:
        query = f"""
            SELECT 
                REVIEW_ID, 
                CITY, 
                RESTAURANT_ID,
                RATING, 
                COMMENT,
                REVIEW_DATE
            FROM ZOMATO.STAGING.STG_REVIEWS
            WHERE COMMENT IS NOT NULL AND TRIM(COMMENT) != ''
            ORDER BY RANDOM()
            LIMIT {limit}
        """
        cur = conn.cursor()
        cur.execute(query)
        df = cur.fetch_pandas_all()
        df.columns = [col.lower() for col in df.columns]
        return df
    finally:
        conn.close()


# ---------------------------------------------------------
# Embedding & Semantic Search Engine
# ---------------------------------------------------------
def get_openai_client():
    if OPENAI_KEY:
        return OpenAI(api_key=OPENAI_KEY, max_retries=1)
    return None


def get_llm_chat_client():
    """Returns (client, model_name, display_name) preferring Gemini if configured."""
    if GEMINI_KEY:
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        client = OpenAI(api_key=GEMINI_KEY, base_url=base_url, timeout=30.0, max_retries=1)
        return client, model, "Google Gemini (gemini-3.5-flash)"
    elif OPENAI_KEY:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=OPENAI_KEY, timeout=30.0, max_retries=0)
        return client, model, f"OpenAI ({model})"
    return None, None, None


def get_openai_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate dense embeddings using OpenAI text-embedding-3-small."""
    client = get_openai_client()
    if not client:
        raise ValueError("OpenAI client not configured")
    # Batch in chunks of 100 for API stability
    all_embeddings = []
    chunk_size = 100
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def compute_tfidf_vectors(corpus: List[str], query: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback TF-IDF vectorizer when OpenAI quota is unavailable."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1536)
    matrix = vectorizer.fit_transform(corpus).toarray()
    query_vec = vectorizer.transform([query]).toarray()[0]
    return matrix, query_vec


@st.cache_data(show_spinner="Syncing and embedding reviews from Snowflake...")
def load_and_prepare_reviews(sample_size: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    """Load reviews from local cache or fetch fresh from Snowflake and embed."""
    if CACHE_FILE.exists():
        try:
            cached_df = pd.read_parquet(CACHE_FILE)
            if len(cached_df) >= min(100, sample_size) and "embedding" in cached_df.columns:
                return cached_df
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}. Re-fetching.")

    df = read_reviews_from_snowflake(limit=sample_size)

    # Attempt dense OpenAI embeddings first
    use_openai = False
    if OPENAI_KEY:
        try:
            embeddings = get_openai_embeddings(df["comment"].tolist())
            df["embedding"] = embeddings
            df["embedding_type"] = "openai"
            use_openai = True
        except Exception as e:
            logger.warning(f"OpenAI embedding unavailable ({e}). Using local TF-IDF semantic index.")

    if not use_openai:
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(stop_words="english", max_features=512)
        tfidf_mat = tfidf.fit_transform(df["comment"].tolist()).toarray()
        df["embedding"] = [v.tolist() for v in tfidf_mat]
        df["embedding_type"] = "tfidf"

    try:
        df.to_parquet(CACHE_FILE)
    except Exception as e:
        logger.warning(f"Could not persist cache: {e}")

    return df


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / norm)


def find_similar_reviews(question: str, df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """Find top-K most semantically relevant reviews for a given question."""
    emb_type = df["embedding_type"].iloc[0] if "embedding_type" in df.columns else "tfidf"

    if emb_type == "openai" and OPENAI_KEY:
        try:
            client = get_openai_client()
            q_res = client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
            q_vec = np.array(q_res.data[0].embedding, dtype=np.float32)
            review_matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
            # Vectorized cosine similarity
            dot_products = np.dot(review_matrix, q_vec)
            norms = np.linalg.norm(review_matrix, axis=1) * np.linalg.norm(q_vec)
            scores = np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms != 0)
        except Exception:
            emb_type = "tfidf"

    if emb_type != "openai":
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = df["comment"].tolist()
        tfidf = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(corpus)
        q_vec = tfidf.transform([question])
        scores = (matrix * q_vec.T).toarray().flatten()

    df_scored = df.copy()
    df_scored["similarity_score"] = scores
    return df_scored.nlargest(top_k, "similarity_score")


# ---------------------------------------------------------
# LLM Generation & Fallback Synthesizer
# ---------------------------------------------------------
def ask_llm(question: str, top_reviews: pd.DataFrame) -> Tuple[str, str]:
    """
    Generate an answer using OpenAI gpt-4o-mini with grounding in retrieved reviews.
    Falls back to a structured analytical summary if OpenAI API quota is exhausted.
    """
    context_lines = []
    for _, row in top_reviews.iterrows():
        context_lines.append(
            f"- [{row['city']} | {row['rating']}★]: \"{row['comment']}\""
        )
    context_str = "\n".join(context_lines)

    client, model, display_name = get_llm_chat_client()
    if client:
        models_to_try = [model]
        if "gemini" in model.lower() and model != "gemini-3.5-flash-lite":
            models_to_try.append("gemini-3.5-flash-lite")
        for m in models_to_try:
            try:
                system_prompt = (
                    "You are an expert restaurant operations and customer experience analyst for Zomato.\n"
                    "Answer the user's question clearly, objectively, and concisely based strictly on the provided customer reviews.\n"
                    "Format your response with clean bullet points and highlight specific customer insights, cities, or common themes.\n"
                    "If the provided reviews do not contain enough information to answer, state that clearly."
                )
                user_prompt = f"User Question: {question}\n\nRetrieved Customer Reviews:\n{context_str}"

                response = client.chat.completions.create(
                    model=m,
                    temperature=0.2,
                    max_tokens=600,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip(), f"Google Gemini ({m})"
            except Exception as e:
                logger.warning(f"Model {m} chat completion error: {e}")

    # Fallback Analytical Synthesizer
    avg_rating = top_reviews["rating"].mean()
    cities = top_reviews["city"].dropna().unique()
    city_str = ", ".join(cities[:4])

    praise_excerpts = top_reviews[top_reviews["rating"] >= 4]["comment"].tolist()
    criticism_excerpts = top_reviews[top_reviews["rating"] <= 2]["comment"].tolist()

    fallback = (
        f"### Review Synthesis for: *\"{question}\"*\n\n"
        f"- **Average Rating of Matching Reviews**: ⭐ **{avg_rating:.1f} / 5.0**\n"
        f"- **Top Locations Mentioned**: {city_str}\n\n"
        "**Key Findings from Retrieved Reviews:**\n"
    )

    if criticism_excerpts:
        fallback += "\n**⚠️ Complaints & Issues:**\n"
        for c in criticism_excerpts[:3]:
            fallback += f"- *\"{c}\"*\n"

    if praise_excerpts:
        fallback += "\n**✨ Highlights & Positive Feedback:**\n"
        for p in praise_excerpts[:3]:
            fallback += f"- *\"{p}\"*\n"

    fallback += (
        "\n\n*(Note: Generated via local semantic ranking engine. Add OpenAI credits to enable live GPT-4o-mini synthesis.)*"
    )
    return fallback, "Local Semantic Engine"


# ---------------------------------------------------------
# Streamlit User Interface
# ---------------------------------------------------------
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Zomato_Logo.svg", width=160)
        st.subheader("Data & Model Control")

        sample_size = st.slider("Snowflake Sample Size", min_value=100, max_value=1000, value=500, step=100)
        top_k = st.slider("Top Matching Reviews (K)", min_value=3, max_value=10, value=TOP_K)

        if st.button("🔄 Refresh Data from Snowflake"):
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown("### Suggested Prompts")
        sample_questions = [
            "What are customers saying about packaging and spills?",
            "How is the delivery partner behaviour and speed?",
            "Are there complaints about high prices or small portions?",
            "What makes customers leave 5-star ratings?",
        ]
        for sq in sample_questions:
            if st.button(sq, key=f"btn_{sq}"):
                st.session_state["pending_prompt"] = sq

    # --- Main Header ---
    st.title("🍔 Zomato AI Review Intelligence")
    st.markdown(
        "Ask natural-language questions across customer reviews stored in **Snowflake**. "
        "The system performs semantic retrieval (RAG) to find matching customer experiences and synthesize answers."
    )

    # --- Load Data ---
    with st.spinner("Connecting to Snowflake and preparing semantic index..."):
        try:
            df = load_and_prepare_reviews(sample_size=sample_size)
        except Exception as e:
            st.error(f"❌ Failed to connect to Snowflake or load reviews: {e}")
            st.info("Check Snowflake credentials in `ai/.env` or `airflow/.env`.")
            return

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Indexed Reviews", f"{len(df):,}")
    m2.metric("Average Rating", f"{df['rating'].mean():.2f} ★")
    m3.metric("Unique Cities", f"{df['city'].nunique()}")
    m4.metric("Search Backend", "OpenAI" if df.get("embedding_type", [""])[0] == "openai" else "TF-IDF Vector")

    st.divider()

    # --- Chat History Management ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your Zomato Customer Review AI Assistant. Ask me anything about delivery speeds, food taste, packaging quality, pricing, or customer complaints across any city!",
                "sources": None,
            }
        ]

    # Display past chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") is not None and not msg["sources"].empty:
                with st.expander(f"📑 View {len(msg['sources'])} Grounding Reviews"):
                    st.dataframe(
                        msg["sources"][["city", "rating", "comment", "similarity_score"]],
                        use_container_width=True,
                        hide_index=True,
                    )

    # --- User Input Handling ---
    user_query = st.chat_input("Ask a question about customer reviews...")
    if "pending_prompt" in st.session_state:
        user_query = st.session_state.pop("pending_prompt")

    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": None})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieval + Generation
        with st.chat_message("assistant"):
            with st.spinner("Searching semantic review vector index..."):
                top_reviews = find_similar_reviews(user_query, df, top_k=top_k)
                answer, model_used = ask_llm(user_query, top_reviews)

            st.markdown(answer)
            with st.expander(f"📑 View {len(top_reviews)} Grounding Reviews ({model_used})"):
                st.dataframe(
                    top_reviews[["city", "rating", "comment", "similarity_score"]],
                    use_container_width=True,
                    hide_index=True,
                )

        # Save assistant message to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": top_reviews}
        )


if __name__ == "__main__":
    main()