"""Streamlit CSV Insight Assistant – rev 7.1 (robust fixes)
==========================================================================
✓ Universal currency→numeric cleaning (no more “$21.70” errors)
✓ Safer PDF generator (adds page, wraps long lines, returns bytes)
✓ Removed leftover heavy‑weight ydata‑profiling call that pulled htmlmin
✓ No Period‑index in MoM chart → convert to str so Plotly can serialise
✓ `tabulate` no longer required — we skip to_markdown calls for speed

Ask free‑form questions; the LLM will call `get_stat()` when it needs hard
numbers.  Click **Generate PDF** to export any answer.
"""
from __future__ import annotations

import os, re, json, textwrap
from typing import List

import pandas as pd
import plotly.express as px
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
    cleaned = (
        s.astype(str)
         .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)   # (123) → -123
         .str.replace(r"[^0-9.\-]", "", regex=True)          # strip $ %, commas
    )
    return pd.to_numeric(cleaned, errors="coerce")

df = pd.read_csv(file)
# apply to obvious currency / numeric‑looking columns
for col in df.columns:
    if df[col].dtype == "object" and df[col].str.contains(r"[0-9].*[0-9]").any():
        df[col] = _to_numeric(df[col])

# detect small‑cardinality categoricals for get_stat
CAT_LIMIT = 150
categoricals: List[str] = [c for c in df.columns if df[c].nunique(dropna=True) <= CAT_LIMIT]

st.dataframe(df.head(), use_container_width=True)

# ── FUNCTION the LLM can call ─────────────────────────────────────────

def get_stat(column: str, value: str, metric: str = "average") -> dict:
    """Return mean / median Rate Variance for rows where column==value."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    mask = df[column].astype(str).str.lower() == value.lower()
    subset = df.loc[mask, "Rate Variance"].dropna()
    if subset.empty:
        return {"error": f"No rows where {column} == {value}"}
    val = subset.mean() if metric == "average" else subset.median()
    return {"rows": int(len(subset)), "metric": metric, "value": float(val)}

fn_spec = {
    "name": "get_stat",
    "description": "Compute average or median Rate Variance for a subset",
    "parameters": {
        "type": "object",
        "properties": {
            "column": {"type": "string"},
            "value":  {"type": "string"},
            "metric": {"type": "string", "enum": ["average", "median"], "default": "average"},
        },
        "required": ["column", "value"],
    },
}

# ── OPENAI CHAT LOOP ──────────────────────────────────────────────────
import openai; openai.api_key = OPENAI_API_KEY
SYSTEM_PROMPT = (
    "You are a senior data analyst with access to the function get_stat. "
    "When users ask for averages, medians, counts etc., call the function."
)

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    st.chat_message(role).markdown(content)

user_q = st.chat_input("Ask about the data… e.g. average variance for affiliate 972")
if user_q:
    st.chat_message("user").markdown(user_q)
    st.session_state.chat.append(("user", user_q))

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": r, "content": c} for r, c in st.session_state.chat
    ]

    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        functions=[fn_spec],
        function_call="auto",
        temperature=0.2,
    )

    choice = resp.choices[0]
    if choice.finish_reason == "function_call":
        args = json.loads(choice.message.function_call.arguments)
        result = get_stat(**args)
        msgs.append({"role": "function", "name": "get_stat", "content": json.dumps(result)})
        final = openai.chat.completions.create(model=OPENAI_MODEL, messages=msgs).choices[0].message.content
    else:
        final = choice.message.content

    st.chat_message("assistant").markdown(final)
    st.session_state.chat.append(("assistant", final))

    # ---------- PDF export ----------
    with st.expander("📄 Export this answer"):
        def _pdf_bytes(title: str, body_md: str) -> bytes:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Helvetica", size=11)
            wrapper = textwrap.TextWrapper(width=100)
            for line in body_md.replace("**", "").splitlines():
                for wrapped in wrapper.wrap(line):
                    pdf.multi_cell(0, 6, wrapped, new_x="LMARGIN", new_y="NEXT")
            pdf.set_title(title)
            return bytes(pdf.output())

        if st.button("Generate PDF", key=f"pdf_{len(st.session_state.chat)}"):
            st.download_button(
                "⬇️ Download PDF", _pdf_bytes("CSV Insight Assistant report", final),
                file_name="analysis.pdf", mime="application/pdf",
                key=f"dl_{len(st.session_state.chat)}",
            )

st.caption("rev 7.1 – robust numeric cleaning & PDF fix")
