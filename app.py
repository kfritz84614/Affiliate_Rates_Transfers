"""Streamlit CSV Insight Assistant – rev 8.0  (generic function‑calling)
===========================================================================
The LLM can now do *any* descriptive analysis without the full CSV:
▶ **aggregate**  – group by a column and compute mean/median/sum/count
▶ **get_rows**   – fetch up to 100 raw rows (selected columns, optional
                   multi‑column filters) so the model can inspect details

The assistant chains calls as it thinks:
    1) aggregate("Affiliate ID", "Rate Variance", "mean", top_n=10)
    2) choose worst affiliate → get_rows(where={...}, columns=[...])

No row‑count or cardinality limits hard‑coded – the LLM decides.  We still
streamline: returns are capped to 100 rows, and aggregates default to top
20 to stay within context.
"""
from __future__ import annotations

import os, re, json, textwrap, ssl
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from fpdf import FPDF

# ── CONFIG ─────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
st.set_page_config("CSV Insight Assistant", page_icon="📊", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Add OPENAI_API_KEY to Secrets.")
    st.stop()

# ── LOAD & CLEAN CSV ───────────────────────────────────────────────────
file = st.file_uploader("Upload CSV", type="csv")
if not file: st.stop()

def to_number(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
         .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
         .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")

df = pd.read_csv(file)
# convert any column that looks numeric but is object
for c in df.columns:
    if df[c].dtype == "object" and df[c].str.contains(r"\d", na=False).any():
        df[c] = to_number(df[c])

st.dataframe(df.head(), use_container_width=True)

# ── FUNCTIONS EXPOSED TO LLM ───────────────────────────────────────────

def aggregate(by: str, target: str, metric: str = "mean", top_n: int = 20) -> Dict:
    """Group by *by* column, compute metric on *target*, return top_n rows."""
    if by not in df.columns or target not in df.columns:
        return {"error": "column not found"}
    if metric not in {"mean", "median", "sum", "count"}:
        return {"error": "invalid metric"}
    series = df.groupby(by)[target]
    if metric == "mean":
        res = series.mean()
    elif metric == "median":
        res = series.median()
    elif metric == "sum":
        res = series.sum()
    else:
        res = series.count()
    tbl = res.sort_values(ascending=False).head(top_n)
    return {"rows": tbl.reset_index().to_dict(orient="records")}

def get_rows(where: Dict[str, str] | None = None, columns: List[str] | None = None, limit: int = 100) -> Dict:
    """Return up to *limit* raw rows after applying simple equality filters."""
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

fns = [
    {
        "name": "aggregate",
        "description": "Group by a column and compute a statistic on target column",
        "parameters": {
            "type": "object",
            "properties": {
                "by": {"type": "string"},
                "target": {"type": "string"},
                "metric": {"type": "string", "enum": ["mean", "median", "sum", "count"], "default": "mean"},
                "top_n": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100}
            },
            "required": ["by", "target"]
        }
    },
    {
        "name": "get_rows",
        "description": "Return raw rows with optional equality filters and selected columns",
        "parameters": {
            "type": "object",
            "properties": {
                "where": {"type": "object"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 100}
            }
        }
    }
]

# ── LLM LOOP ───────────────────────────────────────────────────────────
import openai; openai.api_key = OPENAI_API_KEY
SYSTEM = (
    "You are a senior data analyst. Use the provided functions to answer. "
    "Chain calls as needed and cite numbers in markdown tables."
)
if "chat" not in st.session_state:
    st.session_state.chat = []
for r, m in st.session_state.chat: st.chat_message(r).markdown(m)

q = st.chat_input("Ask anything about the dataset…")
if q:
    st.chat_message("user").markdown(q)
    st.session_state.chat.append(("user", q))

    msgs = [{"role": "system", "content": SYSTEM}] + [
        {"role": r, "content": m} for r, m in st.session_state.chat
    ]

    while True:
        resp = openai.chat.completions.create(model=OPENAI_MODEL, messages=msgs, functions=fns, function_call="auto")
        msg = resp.choices[0].message
        if msg.function_call:
            fn_name = msg.function_call.name
            args = json.loads(msg.function_call.arguments or "{}")
            if fn_name == "aggregate":
                result = aggregate(**args)
            elif fn_name == "get_rows":
                result = get_rows(**args)
            else:
                result = {"error": "unknown function"}
            msgs.append({"role": "function", "name": fn_name, "content": json.dumps(result)})
            # loop and let model craft final answer after seeing function result
            continue
        else:
            answer = msg.content.strip()
            break

    st.chat_message("assistant").markdown(answer)
    st.session_state.chat.append(("assistant", answer))

st.caption("rev 8.0 – generic aggregate / get_rows function‑calling")
