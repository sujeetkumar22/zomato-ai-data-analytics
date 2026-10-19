"""
Enrich Reviews Script for Zomato AI Data Analytics Pipeline
===========================================================
This module handles automated sentiment analysis and aspect extraction for restaurant
reviews stored in Snowflake.

Workflow:
1. Connect to Snowflake (ZOMATO database, RAW and AI schemas).
2. Ensure destination table `ZOMATO.AI.ENRICHED_REVIEWS` exists.
3. Query unprocessed reviews from `ZOMATO.RAW.REVIEWS`.
4. Enrich each review with:
   - Sentiment (Positive / Neutral / Negative)
   - Sentiment Score (-1.0 to 1.0)
   - Primary Aspect (Food Quality, Delivery Time, Packaging, Pricing, Service, etc.)
   - Category Tags (JSON array)
   - Key Issue or Highlight summary
   - Complaint flag (True / False)
   Using OpenAI (gpt-4o-mini) when OPENAI_API_KEY is configured, with an intelligent
   heuristic fallback if the API key is not yet set.
5. Batch insert the enriched reviews into `ZOMATO.AI.ENRICHED_REVIEWS`.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import snowflake.connector
from dotenv import load_dotenv

# Set up clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("enrich_reviews")

# Load environment variables from local .env, airflow/.env, and project root
_curr = Path(__file__).resolve().parent
load_dotenv(_curr / ".env")
load_dotenv(_curr.parent / "airflow" / ".env")
load_dotenv(_curr.parent / ".env")


def get_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """Establish and return a Snowflake database connection."""
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    database = os.getenv("SNOWFLAKE_DATABASE", "ZOMATO")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "ZOMATO_WH")
    role = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")

    if not password:
        raise ValueError("SNOWFLAKE_PASSWORD is not set in environment or .env file.")

    logger.info(f"Connecting to Snowflake account='{account}', user='{user}', db='{database}'...")
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        role=role,
    )
    return conn


def init_destination_table(conn: snowflake.connector.SnowflakeConnection) -> None:
    """Ensure the AI schema and ENRICHED_REVIEWS table exist in Snowflake."""
    ddl_schema = "CREATE SCHEMA IF NOT EXISTS ZOMATO.AI"
    ddl_table = """
    CREATE TABLE IF NOT EXISTS ZOMATO.AI.ENRICHED_REVIEWS (
        REVIEW_ID NUMBER(38, 0),
        ORDER_ID NUMBER(38, 0),
        USER_ID NUMBER(38, 0),
        RESTAURANT_ID NUMBER(38, 0),
        RATING NUMBER(38, 0),
        COMMENT VARCHAR(16777216),
        REVIEW_DATE DATE,
        SENTIMENT VARCHAR(20),
        SENTIMENT_SCORE FLOAT,
        PRIMARY_ASPECT VARCHAR(50),
        TAGS VARCHAR(16777216),
        KEY_ISSUE_OR_HIGHLIGHT VARCHAR(16777216),
        IS_COMPLAINT BOOLEAN,
        ENRICHED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
    )
    """
    with conn.cursor() as cur:
        cur.execute(ddl_schema)
        cur.execute(ddl_table)
    logger.info("Destination table 'ZOMATO.AI.ENRICHED_REVIEWS' verified / initialized.")


def fetch_pending_reviews(
    conn: snowflake.connector.SnowflakeConnection, limit: int
) -> List[Tuple[Any, ...]]:
    """
    Fetch un-enriched reviews from ZOMATO.RAW.REVIEWS.
    Excludes reviews that are already present in ZOMATO.AI.ENRICHED_REVIEWS.
    """
    query = """
    SELECT 
        r.REVIEW_ID,
        r.ORDER_ID,
        r.USER_ID,
        r.RESTAURANT_ID,
        r.RATING,
        r.COMMENT,
        r.REVIEW_DATE
    FROM ZOMATO.RAW.REVIEWS r
    WHERE r.COMMENT IS NOT NULL 
      AND TRIM(r.COMMENT) != ''
      AND r.REVIEW_ID NOT IN (SELECT REVIEW_ID FROM ZOMATO.AI.ENRICHED_REVIEWS)
    ORDER BY r.REVIEW_DATE DESC
    LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (limit,))
        rows = cur.fetchall()
    return rows


def analyze_with_llm(
    comment: str, rating: int, client: Any, model: str
) -> Optional[Dict[str, Any]]:
    """Analyze review comment using LLM Chat Completions API with JSON mode."""
    try:
        system_prompt = (
            "You are an expert NLP data annotator for restaurant reviews on Zomato.\n"
            "Analyze the given customer review text and numeric rating (1-5).\n"
            "Choose sentiment from: Positive, Neutral, Negative.\n"
            "Choose primary_aspect from: Food Quality, Delivery Time, Packaging, Price, Customer Service, Hygiene, Other.\n"
            "Respond ONLY with a valid JSON object formatted exactly as:\n"
            "{\n"
            '  "sentiment": "Positive",\n'
            '  "sentiment_score": 0.8,\n'
            '  "primary_aspect": "Food Quality",\n'
            '  "tags": ["fresh", "delicious"],\n'
            '  "key_issue_or_highlight": "Great food taste",\n'
            '  "is_complaint": false\n'
            "}"
        )
        user_prompt = f"Customer Rating: {rating}/5\nReview Text: \"{comment}\""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600,
        )

        content = response.choices[0].message.content
        if content:
            raw = content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            # Find JSON boundaries if extra text exists
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
            # Remove trailing commas before closing braces/brackets
            raw = re.sub(r',\s*([\]}])', r'\1', raw)
            try:
                return json.loads(raw)
            except Exception:
                import ast
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
                raise
    except Exception as e:
        logger.warning(f"LLM analysis error for comment '{comment[:30]}...': {e}")
    return None


def heuristic_enrichment(comment: str, rating: int) -> Dict[str, Any]:
    """
    Intelligent heuristic fallback used when OpenAI API key is unavailable or fails.
    Classifies sentiment, aspects, and complaints using rating signals and keyword matching.
    """
    lower = comment.lower()

    # Aspect keyword mapping
    aspect_patterns = {
        "Packaging": ["package", "packaging", "spill", "leak", "box", "container", "bag", "eco-friendly"],
        "Delivery Time": ["late", "delay", "on time", "rider", "driver", "cold", "fast", "speed", "delivery"],
        "Food Quality": ["taste", "fresh", "stale", "oily", "cooked", "delicious", "flavor", "salty", "raw", "burnt"],
        "Price": ["expensive", "price", "overpriced", "worth", "cheap", "cost", "value"],
        "Customer Service": ["polite", "rude", "service", "helpful", "behaviour", "attitude", "staff"],
    }

    detected_aspects = []
    for aspect, keywords in aspect_patterns.items():
        if any(kw in lower for kw in keywords):
            detected_aspects.append(aspect)

    primary_aspect = detected_aspects[0] if detected_aspects else "Food Quality"

    # Determine sentiment and complaint
    complaint_words = ["spill", "stale", "oily", "tasteless", "late", "expensive", "rude", "bad", "worst", "cold"]
    is_complaint = rating <= 2 or any(w in lower for w in complaint_words)

    if rating >= 4:
        sentiment = "Positive"
        sentiment_score = 0.5 + (rating - 3) * 0.25
    elif rating == 3:
        sentiment = "Neutral"
        sentiment_score = 0.0
    else:
        sentiment = "Negative"
        sentiment_score = -0.5 - (3 - rating) * 0.25

    # Refine key highlight / issue
    if is_complaint:
        key_issue = f"Complaint: {comment[:50].strip()}"
    else:
        key_issue = f"Highlight: {comment[:50].strip()}"

    tags = detected_aspects if detected_aspects else ["general_feedback"]

    return {
        "sentiment": sentiment,
        "sentiment_score": round(sentiment_score, 2),
        "primary_aspect": primary_aspect,
        "tags": tags,
        "key_issue_or_highlight": key_issue,
        "is_complaint": is_complaint,
    }


def enrich_review(
    comment: str, rating: int, client: Optional[Any], model: Optional[str]
) -> Dict[str, Any]:
    """Enrich a single review, attempting LLM first then falling back to heuristics."""
    if client and model:
        result = analyze_with_llm(comment, rating, client, model)
        if result:
            return result

    # Fallback to intelligent heuristic
    return heuristic_enrichment(comment, rating)


def save_enriched_reviews(
    conn: snowflake.connector.SnowflakeConnection, records: List[Tuple[Any, ...]]
) -> int:
    """Insert enriched records into ZOMATO.AI.ENRICHED_REVIEWS."""
    insert_sql = """
    INSERT INTO ZOMATO.AI.ENRICHED_REVIEWS (
        REVIEW_ID,
        ORDER_ID,
        USER_ID,
        RESTAURANT_ID,
        RATING,
        COMMENT,
        REVIEW_DATE,
        SENTIMENT,
        SENTIMENT_SCORE,
        PRIMARY_ASPECT,
        TAGS,
        KEY_ISSUE_OR_HIGHLIGHT,
        IS_COMPLAINT
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cur:
        cur.executemany(insert_sql, records)
        conn.commit()
    return len(records)


def main() -> None:
    """Main execution function for the review enrichment pipeline."""
    logger.info("=== Starting Zomato Review AI Enrichment Pipeline ===")

    # Configuration
    sample_n = int(os.getenv("SAMPLE_N", 5))
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    llm_client = None
    llm_model = None

    if gemini_key:
        try:
            from openai import OpenAI
            base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
            llm_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
            llm_client = OpenAI(api_key=gemini_key, base_url=base_url, timeout=30.0, max_retries=1)
            logger.info(f"Using Google Gemini model '{llm_model}' for sentiment enrichment.")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini client: {e}")
    elif openai_key:
        try:
            from openai import OpenAI
            llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            llm_client = OpenAI(api_key=openai_key, timeout=30.0, max_retries=0)
            logger.info(f"Using OpenAI model '{llm_model}' for sentiment enrichment.")
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI client: {e}")

    if not llm_client:
        logger.warning(
            "No active LLM API key available. Using intelligent heuristic fallback "
            "(Rating + Aspect NLP classifier)."
        )

    # 1. Connect to Snowflake
    conn = get_snowflake_connection()
    try:
        # 2. Ensure schema & destination table exist
        init_destination_table(conn)

        # 3. Fetch un-enriched reviews
        logger.info(f"Querying up to {sample_n} pending reviews from ZOMATO.RAW.REVIEWS...")
        raw_reviews = fetch_pending_reviews(conn, sample_n)

        if not raw_reviews:
            logger.info("No pending reviews found to enrich. Everything is up to date!")
            return

        logger.info(f"Found {len(raw_reviews)} reviews to process.")

        # 4. Enrich reviews
        enriched_records = []
        for row in raw_reviews:
            review_id, order_id, user_id, restaurant_id, rating, comment, review_date = row

            enrichment = enrich_review(comment, rating, llm_client, llm_model)

            tags_str = json.dumps(enrichment.get("tags", []))
            record = (
                review_id,
                order_id,
                user_id,
                restaurant_id,
                rating,
                comment,
                review_date,
                enrichment.get("sentiment", "Neutral"),
                float(enrichment.get("sentiment_score", 0.0)),
                enrichment.get("primary_aspect", "Other"),
                tags_str,
                enrichment.get("key_issue_or_highlight", "")[:255],
                bool(enrichment.get("is_complaint", False)),
            )
            enriched_records.append(record)

            logger.info(
                f"Review #{review_id} (Rating {rating}/5): "
                f"Sentiment={record[7]}, Aspect='{record[9]}', Complaint={record[12]}"
            )

        # 5. Insert into Snowflake
        inserted_count = save_enriched_reviews(conn, enriched_records)
        logger.info(f"Successfully wrote {inserted_count} enriched reviews into ZOMATO.AI.ENRICHED_REVIEWS.")

    finally:
        conn.close()
        logger.info("Snowflake connection closed.")
        logger.info("=== Zomato Review AI Enrichment Pipeline Completed ===")


if __name__ == "__main__":
    main()
