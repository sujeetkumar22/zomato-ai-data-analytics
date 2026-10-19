import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
_curr = Path(__file__).resolve().parent
load_dotenv(_curr / ".env")
load_dotenv(_curr.parent / "airflow" / ".env")
load_dotenv(_curr.parent / ".env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

FORBIDDEN_WORDS = [
    'drop', 'delete', 'truncate', 'alter', 'update', 'insert',
    'create', 'replace', 'grant', 'revoke'
]

EXAMPLE_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisin has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method",
    "Top 10 restaurants by revenue in Bangalore",
]

SCHEMA = """
Tables available (Snowflake). Use bare table names, no database or schema prefix.
 
FCT_ORDERS(order_id, order_date, customer_id, restaurant_id, city, cuisine,
           payment_method, order_status, is_delivered, sales_amount, discount,
           delivery_fee, gst, customer_rating, delivery_time_min)
DIM_RESTAURANT(restaurant_id, restaurant_name, city, cuisine, rating, cost_for_two)
DIM_CUSTOMER(customer_id, customer_name, age, age_segment, gender, city)
MART_DAILY_CITY_REVENUE(order_date, city, orders, cancel_rate, gmv, aov)
MART_RESTAURANT_PERFORMANCE(restaurant_id, restaurant_name, city, cuisine,
                            orders, revenue, avg_customer_rating, cancel_rate)
MART_DELIVE_SLA(city, order_hour, delivered_orders, p50, p90)

Note: gmv means delivered revenue (sales_amount where is_delivered = true). 
Prefer the MART_ tables when they fit the question.
"""

SYSTEM_PROMPT = f"""
You are a Snowflake SQL expert. Write ONE SELECT query that answers the question.
 
Rules:
- SELECT queries only, never modify data.
- Use bare table names (FCT_ORDERS, not ZOMATO.MARTS.FCT_ORDERS).
- Add a LIMIT of 100 or less, unless the question asks for a single total.
- Reply as JSON in this exact format: {{"sql": "your query here"}}
 
{SCHEMA}
"""


@st.cache_resource
def get_connection():
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "ZOMATO_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "ZOMATO"),
        schema="MARTS",
        role=os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    )


def generate_fallback_sql(question: str) -> str:
    """Fallback SQL generator when OpenAI quota is unavailable."""
    q = question.lower()
    if "city" in q and ("gmv" in q or "revenue" in q or "sales" in q):
        return (
            "SELECT city, ROUND(SUM(IFF(is_delivered, sales_amount, 0)), 2) AS total_gmv\n"
            "FROM FCT_ORDERS\n"
            "WHERE city IS NOT NULL\n"
            "GROUP BY city\n"
            "ORDER BY total_gmv DESC\n"
            "LIMIT 10"
        )
    if "cuisin" in q or "cuisine" in q:
        return (
            "SELECT cuisine, COUNT(*) AS total_orders\n"
            "FROM FCT_ORDERS\n"
            "WHERE cuisine IS NOT NULL\n"
            "GROUP BY cuisine\n"
            "ORDER BY total_orders DESC\n"
            "LIMIT 10"
        )
    if "delivery" in q and ("time" in q or "worst" in q or "slow" in q or "speed" in q):
        return (
            "SELECT city, ROUND(AVG(delivery_time_min), 1) AS avg_delivery_time_min\n"
            "FROM FCT_ORDERS\n"
            "WHERE is_delivered AND delivery_time_min IS NOT NULL AND city IS NOT NULL\n"
            "GROUP BY city\n"
            "HAVING COUNT(*) >= 50\n"
            "ORDER BY avg_delivery_time_min DESC\n"
            "LIMIT 10"
        )
    if "cancel" in q or "payment" in q:
        return (
            "SELECT \n"
            "    payment_method,\n"
            "    COUNT(*) AS total_orders,\n"
            "    ROUND(DIV0(COUNT_IF(order_status = 'Cancelled'), COUNT(*)) * 100, 2) AS cancel_rate_pct\n"
            "FROM FCT_ORDERS\n"
            "WHERE payment_method IS NOT NULL\n"
            "GROUP BY payment_method\n"
            "ORDER BY cancel_rate_pct DESC\n"
            "LIMIT 10"
        )
    if "restaurant" in q and ("revenue" in q or "bangalore" in q or "top" in q):
        return (
            "SELECT restaurant_name, revenue, orders\n"
            "FROM MART_RESTAURANT_PERFORMANCE\n"
            "WHERE LOWER(city) = 'bangalore'\n"
            "ORDER BY revenue DESC\n"
            "LIMIT 10"
        )
    return (
        "SELECT city, COUNT(*) AS orders, ROUND(SUM(IFF(is_delivered, sales_amount, 0)), 2) AS gmv\n"
        "FROM FCT_ORDERS\n"
        "WHERE city IS NOT NULL\n"
        "GROUP BY city\n"
        "ORDER BY gmv DESC\n"
        "LIMIT 10"
    )


def get_llm_client():
    """Returns (client, model_name, display_name) preferring Gemini if configured."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if gemini_key:
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        client = OpenAI(api_key=gemini_key, base_url=base_url, timeout=30.0, max_retries=1)
        return client, model, "Google Gemini (gemini-3.5-flash)"
    elif openai_key:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=openai_key, timeout=30.0, max_retries=0)
        return client, model, f"OpenAI ({model})"
    return None, None, None


def generate_sql(question: str) -> tuple:
    client, model, display_name = get_llm_client()
    if client:
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]
            )
            raw = response.choices[0].message.content.strip()
            # Clean possible markdown wrap
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            data = json.loads(raw)
            sql = data.get("sql", raw)
            # Remove any database or schema prefix for Snowflake MARTS schema
            sql = re.sub(r'\bZOMATO\.(MARTS|STAGING|AI)\.', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bZOMATO\.', '', sql, flags=re.IGNORECASE)
            return sql.strip().rstrip(";"), display_name
        except Exception as e:
            st.warning(f"⚠️ {display_name} API call notice: {e}. Utilizing rule-based schema fallback.")
            return generate_fallback_sql(question), "Rule-Based Fallback"
    else:
        st.info("ℹ️ No LLM API key configured. Using rule-based schema SQL generator.")
        return generate_fallback_sql(question), "Rule-Based Fallback"


def is_safe(sql: str) -> bool:
    lowered = sql.lower()

    if not lowered.startswith("select") and not lowered.startswith("with"):
        return False

    for word in FORBIDDEN_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', lowered):
            return False

    return True


def run_query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    cursor = conn.cursor()
    return cursor.execute(sql).fetch_pandas_all()


_, _, active_model_display = get_llm_client()
if not active_model_display:
    active_model_display = "Rule-Based Fallback"

st.title("🍔 Chat with your Zomato Data")
st.caption(f"Ask questions in natural English — powered by **{active_model_display}** & **Snowflake** (10M Rows)")

with st.sidebar:
    st.header("⚡ Example Questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"btn_{q}"):
            st.session_state["selected_question"] = q

prefilled = st.session_state.pop("selected_question", "")
question = st.text_input(
    "Enter your question here", 
    value=prefilled,
    placeholder="e.g. Top 10 restaurants by revenue in Bangalore"
)

if question:
    sql, engine_used = generate_sql(question)
    st.markdown(f"**Generated SQL Query** *(via {engine_used})*:")
    st.code(sql, language="sql")

    if not is_safe(sql):
        st.error("The generated SQL is not safe to run. Please modify your question.")
    else:
        try:
            with st.spinner("Executing query on Snowflake..."):
                df = run_query(sql)
            st.success(f"{len(df):,} rows returned from Snowflake")
            st.dataframe(df, hide_index=True)

            if len(df.columns) >= 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                st.bar_chart(df, x=df.columns[0], y=df.columns[1])

        except Exception as e:
            st.error(f"Error running query: {e}")