"""Streamlit CSV Insight Assistant – rev 8.2 (headline + robust SYSTEM)
===========================================================================
* Restores headline KPI block (total / mean / median / overrun share).
* SYSTEM prompt now embeds numeric `describe()` once (no duplicate).
* Keeps generic `aggregate` / `get_rows` function‑calling flow.
"""
from __future__ import annotations

import os, re, json, textwrap
from io import BytesIO
from typing import Dict, List

import pandas as pd
import streamlit as st
from fpdf import FPDF
import openai

# ── CONFIG ─────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
st.set_page_config("CSV Insight Assistant", page_icon="📊", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Add OPENAI_API_KEY to Secrets ⇒ Settings → Secrets")
    st.stop()
openai.api_key = OPENAI_API_KEY

# ── LOAD + CLEAN CSV ───────────────────────────────────────────────────
file = st.file_uploader("Upload CSV", type="csv")
if not file:
    st.stop()

def to_number(s: pd.Series) -> pd.Series:
    cleaned = (s.astype(str)
                 .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
                 .str.replace(r"[^0-9.\-]", "", regex=True))
    return pd.to_numeric(cleaned, errors="coerce")

df = pd.read_csv(file)
for c in df.columns:
    if df[c].dtype == "object" and df[c].str.contains(r"\d", na=False).any():
        df[c] = to_number(df[c])

st.dataframe(df.head(), use_container_width=True)

# ── HEADLINE KPIs ───────────────────────────────────────────────────────
if "Rate Variance" in df.columns:
    rv = df["Rate Variance"].dropna(); over = rv[rv > 0]
    headline = {
        "Total net variance": f"${rv.sum():,.0f}",
        "Average variance": f"${rv.mean():,.2f}",
        "Median variance": f"${rv.median():,.2f}",
        "Jobs with overrun": f"{len(over):,} ({len(over)/len(df):.1%})",
    }
    st.subheader("📌 Headline summary")
    st.table(pd.DataFrame(headline, index=["Value"]))

# ── FUNCTION DEFINITIONS ────────────────────────────────────────────────

def aggregate(by: str, target: str, metric: str = "mean", top_n: int = 20) -> Dict:
    """Group by *by*, compute metric on *target* and return top_n rows.
    Safely avoids duplicate-column error when *by* == *target* by naming the
    value column "Result".
    """
    if by not in df.columns or target not in df.columns:
        return {"error": "column not found"}
    series = df.groupby(by)[target]
    func = {"mean": series.mean, "median": series.median, "sum": series.sum, "count": series.count}[metric]
    tbl = func().sort_values(ascending=False).head(top_n)
    tbl = tbl.rename("Result").reset_index()  # avoids duplicate col name
    return {"rows": tbl.to_dict(orient="records")} {"error": "column not found"}
    series = df.groupby(by)[target]
    func = {"mean": series.mean, "median": series.median, "sum": series.sum, "count": series.count}[metric]
    tbl = func().sort_values(ascending=False).head(top_n)
    return {"rows": tbl.reset_index().to_dict(orient="records")}

def get_rows(where: Dict[str, str] | None = None, columns: List[str] | None = None, limit: int = 100) -> Dict:
    sub = df
    if where:
        for col, val in where.items():
            if col not in df.columns:
                return {"error": f"column {col} not found"}
            sub = sub[sub[col].astype(str) == str(val)]
    if columns:
        for col in columns:
            if col not in df.columns:
                return {"error": f"column {col} not found"}
        sub = sub[columns]
    return {"rows": sub.head(limit).to_dict(orient="records")}

FUNCTIONS = [
    {"name": "aggregate", "description": "Group by column and compute statistic", "parameters": {
        "type": "object", "properties": {
            "by": {"type": "string"}, "target": {"type": "string"},
            "metric": {"type": "string", "enum": ["mean","median","sum","count"], "default": "mean"},
            "top_n": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100}
        }, "required": ["by", "target"]}},
    {"name": "get_rows", "description": "Return raw rows after filters", "parameters": {
        "type": "object", "properties": {
            "where": {"type": "object"}, "columns": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 100}
        }}}
]

# ── CHAT SESSION ────────────────────────────────────────────────────────
if "chat" not in st.session_state: st.session_state.chat = []
for role, content in st.session_state.chat: st.chat_message(role).markdown(content)

numeric_md = df.describe(include="number").to_markdown()
SYSTEM_PROMPT = (
    "You are a senior data analyst. Here is a numeric summary of the dataset:\n" + numeric_md +
    "\nUse the functions aggregate() and get_rows() to answer user questions. "
    "When you call a function, wait for its result before replying." )

q = st.chat_input("Ask anything about the data…")
if q:
    st.chat_message("user").markdown(q)
    st.session_state.chat.append(("user", q))

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": r, "content": c} for r, c in st.session_state.chat
    ]

    while True:
        resp = openai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, functions=FUNCTIONS, function_call="auto", temperature=0.2)
        msg = resp.choices[0].message
        if msg.function_call:
            fn_name = msg.function_call.name
            args = json.loads(msg.function_call.arguments or "{}")
            result = aggregate(**args) if fn_name == "aggregate" else get_rows(**args)
            msgs.append({"role": "function", "name": fn_name, "content": json.dumps(result)})
            continue  # loop again so model can craft final answer
        answer = msg.content.strip(); break

    st.chat_message("assistant").markdown(answer)
    st.session_state.chat.append(("assistant", answer))

st.caption("rev 8.2 – headline restored; numeric summary in system prompt")
