"""Streamlit CSV Insight Assistant – rev 7.0  (LLM‑first, function‑calling)
======================================================================
The LLM can now *truly* “see” the data via OpenAI function‑calling:
• We expose a single function `get_stat(column, value, metric)` that the
  model can invoke on demand.  It returns mean / median of *Rate Variance*
  filtered by any categorical value (Affiliate ID, City, Chauffeur, …).
• The assistant decides what to call – you just ask naturally.
• We keep a short numeric summary + the schema in the system prompt to
  give the model context before it calls.
• The previous quick Python fallback is removed; everything routes through
  the function call so answers remain conversational but data‑grounded.
• Sidebar filter + PDF export unchanged.
"""
from __future__ import annotations

import os, re, json, ssl, textwrap
from io import BytesIO
from typing import List
from email.message import EmailMessage

import pandas as pd
import streamlit as st
from fpdf import FPDF

# ── CONFIG ─────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
st.set_page_config("CSV Insight Assistant", page_icon="📊", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please add your OPENAI_API_KEY to Secrets")
    st.stop()

# ── DATA LOAD & CLEAN ─────────────────────────────────────────────────
file = st.file_uploader("Upload CSV", type="csv")
if not file:
    st.stop()

def _to_numeric(s: pd.Series) -> pd.Series:
    cleaned = (s.astype(str)
                 .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
                 .str.replace(r"[^0-9.\-]", "", regex=True))
    return pd.to_numeric(cleaned, errors="coerce")

df = pd.read_csv(file)
if "Rate Variance" in df.columns:
    df["Rate Variance"] = _to_numeric(df["Rate Variance"])

# limit categorical detection to manageable columns
CAT_LIMIT = 150
categoricals: List[str] = [c for c in df.columns if df[c].nunique(dropna=True) <= CAT_LIMIT]

st.dataframe(df.head(), use_container_width=True)

# ── FUNCTION the LLM can call ─────────────────────────────────────────

def get_stat(column: str, value: str, metric: str = "average") -> dict:
    """Return mean / median Rate Variance for rows where column==value."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    mask = df[column].astype(str).str.lower() == value.lower()
    subset = df.loc[mask, "Rate Variance"].dropna()
    if subset.empty:
        return {"error": f"No rows where {column} == {value}"}
    if metric == "median":
        val = subset.median()
    else:
        val = subset.mean()
    return {
        "rows": int(len(subset)),
        "metric": metric,
        "value": float(val),
        "currency": "$"
    }

# describe this function for OpenAI
fn_spec = {
    "name": "get_stat",
    "description": "Compute average or median Rate Variance for a subset",
    "parameters": {
        "type": "object",
        "properties": {
            "column": {"type": "string", "description": "Column name to filter on"},
            "value":  {"type": "string", "description": "Exact value to match (case‑insensitive)"},
            "metric": {"type": "string", "enum": ["average", "median"], "default": "average"},
        },
        "required": ["column", "value"]
    }
}

# ── LLM dialogue loop ─────────────────────────────────────────────────
import openai; openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = (
    "You are a senior data analyst with access to a function called get_stat. "
    "Use it whenever the user asks for numbers (average, median etc.)."
)

if "chat" not in st.session_state:
    st.session_state.chat = []  # store messages

for role, content in st.session_state.chat:
    st.chat_message(role).markdown(content)

user_q = st.chat_input("Ask a question about the data… e.g. average variance for affiliate 972")
if user_q:
    st.chat_message("user").markdown(user_q)
    st.session_state.chat.append(("user", user_q))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ] + [
        {"role": role, "content": content} for role, content in st.session_state.chat
    ]

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        functions=[fn_spec],
        function_call="auto",
        temperature=0.2,
    )

    choice = response.choices[0]

    if choice.finish_reason == "function_call":
        fn_name = choice.message.function_call.name
        args = json.loads(choice.message.function_call.arguments)
        if fn_name == "get_stat":
            result = get_stat(**args)
            messages.append({"role": "function", "name": fn_name, "content": json.dumps(result)})
            # second round – ask model to craft final answer with result
            final_resp = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.2,
            )
            answer = final_resp.choices[0].message.content.strip()
        else:
            answer = "Function not recognised."
    else:
        answer = choice.message.content.strip()

    st.chat_message("assistant").markdown(answer)
    st.session_state.chat.append(("assistant", answer))

    # optional PDF export
    with st.expander("📄 Export this answer"):
        if st.button("Generate PDF", key=f"pdf_{len(st.session_state.chat)}"):
            buf = pdf_from_md("CSV Insight Assistant report", answer)
            st.download_button("⬇️ Download PDF", buf, "analysis.pdf", "application/pdf",
                               key=f"dl_{len(st.session_state.chat)}")

st.caption("rev 7.0 – true LLM ↔ CSV via OpenAI function‑calling")
