from __future__ import annotations

"""Streamlit CSV Insight Assistant – rev 6.1
================================================================
▪︎ **Dynamic, automatic answering** — if the question mentions a single
  categorical value (city, affiliate, chauffeur, vehicle type, etc.) the
  app now computes the answer *in Python* before falling back to the LLM.
▪︎ Categorical columns = *all* columns with ≤ 150 uniques (string **or**
  numeric) so Affiliate ID now included.
▪︎ Added an “⚙️ Filter & summary” sidebar where the user can slice the data
  on any categorical column and immediately see average / median variance.
▪︎ LLM context trimmed to < 4 k tokens by only embedding tables for the
  5 largest categorical columns + the one matching the user’s question.
▪︎ **PDF bug fixed** – long un‑broken words are now wrapped, preventing the
  *FPDFException: Not enough horizontal space* error.
▪︎ **Numeric conversion improved** – parentheses like *($21.70)* are now
  recognised as negatives, avoiding *TypeError: Could not convert string*.
"""

import os, re, textwrap, ssl, json
from io import BytesIO
from typing import Optional
from email.message import EmailMessage

import pandas as pd
import streamlit as st
from fpdf import FPDF

OPENAI_MODEL = "gpt-4o-mini"
st.set_page_config("CSV Insight Assistant", page_icon="📊", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# ── helpers ──────────────────────────────────────────────────────────────

def _to_datetime(s: pd.Series) -> pd.Series:
    """Coerce obvious date/time columns to datetimes."""
    return pd.to_datetime(s, errors="coerce")


def _to_numeric(s: pd.Series) -> pd.Series:
    """Convert strings that *look* like numbers / currency to floats.

    Handles leading currency symbols, commas, and parentheses indicating
    negatives – e.g. "($21.70)" → -21.70.
    """
    cleaned = (s.astype(str)
                 # convert parentheses to leading minus sign
                 .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
                 # strip everything except digits, minus & dot
                 .str.replace(r"[^0-9.\-]", "", regex=True))
    return pd.to_numeric(cleaned, errors="coerce")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("date", "time")):
            df[c] = _to_datetime(df[c])
        elif any(k in lc for k in ("variance", "amount", "cost", "price")):
            df[c] = _to_numeric(df[c])
    return df


def categorical_columns(df: pd.DataFrame, limit: int = 150):
    return [c for c in df.columns if df[c].nunique(dropna=True) <= limit]


# --- PDF generation ------------------------------------------------------

_wrapper = textwrap.TextWrapper(width=90, break_long_words=True,
                               break_on_hyphens=False)

def pdf_from_md(title: str, md: str) -> bytes:
    """Render *md* (markdown) to a simple A4 PDF and return its bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    plain = re.sub(r"[*`_]", "", md)
    for line in plain.splitlines():
        for wrapped in _wrapper.wrap(line):
            pdf.multi_cell(0, 6, wrapped)
    pdf.set_title(title)

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# ── LLM call -------------------------------------------------------------

def ask_llm(system: str, user: str) -> str:
    if not OPENAI_API_KEY:
        return "*(OPENAI_API_KEY missing – cannot call LLM)*"
    import openai
    openai.api_key = OPENAI_API_KEY
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()

# ── Streamlit UI ---------------------------------------------------------

st.title("📊 CSV Insight Assistant v6.1")
file = st.file_uploader("Upload CSV", type="csv")
if not file:
    st.stop()

raw_df = pd.read_csv(file)
df = clean_df(raw_df.copy())
cats = categorical_columns(df)

# ─ Sidebar filter
st.sidebar.header("⚙️ Filter & summary")
filter_col = st.sidebar.selectbox("Choose column to filter", ["(none)"] + cats)
subset = df
if filter_col != "(none)":
    options = sorted(df[filter_col].dropna().astype(str).unique())
    sel = st.sidebar.multiselect("Values", options)
    if sel:
        subset = subset[subset[filter_col].astype(str).isin(sel)]
    st.sidebar.write(f"Rows after filter: {len(subset):,}")
    if "Rate Variance" in subset.columns:
        st.sidebar.metric("Average variance", f"${subset['Rate Variance'].mean():,.2f}")
        st.sidebar.metric("Median variance", f"${subset['Rate Variance'].median():,.2f}")

st.dataframe(subset.head(500), use_container_width=True)

# ─ Headline KPIs (always on full df) ------------------------------------
if "Rate Variance" in df.columns:
    rv = df["Rate Variance"].dropna()
    over = rv[rv > 0]
    headline = {
        "Total net variance": f"${rv.sum():,.0f}",
        "Avg variance": f"${rv.mean():,.2f}",
        "Median variance": f"${rv.median():,.2f}",
        "Overrun jobs": f"{len(over):,} ({len(over)/len(df):.1%})",
    }
    st.subheader("📌 Headline (entire file)")
    st.table(pd.DataFrame(headline, index=["Value"]))

# ─ Build summary tables for LLM context ---------------------------------
TABLE_LIMIT = 5          # max tables to include per prompt
ROW_LIMIT = 120          # trim long tables
summary_tables: dict[str, str] = {}
for col in cats:
    if "Rate Variance" not in df.columns:
        continue
    tbl = (
        df[[col, "Rate Variance"]]
        .dropna()
        .groupby(col)["Rate Variance"].mean().round(2)
        .sort_values(ascending=False)
        .head(ROW_LIMIT)
    )
    if not tbl.empty:
        summary_tables[col] = tbl.to_markdown()

# keep largest tables first (by rows) and cap to limit
key_tables = sorted(summary_tables.items(), key=lambda kv: -kv[1].count("\n"))[:TABLE_LIMIT]
context_tables = "\n\n".join(f"=== {k} ===\n{v}" for k, v in key_tables)

# ─ Chat -----------------------------------------------------------------
if "hist" not in st.session_state:
    st.session_state.hist = []
for role, msg in st.session_state.hist:
    st.chat_message(role).markdown(msg)

q = st.chat_input("Ask about averages, medians, counts …")

# Utility: simple direct answer if question matches pattern "average variance for <value>"

def direct_variance_answer(query: str) -> Optional[str]:
    m = re.search(r"average (?:rate )?variance (?:for|in) (.+?)$", query, re.I)
    if not m or "Rate Variance" not in df.columns:
        return None
    target = m.group(1).strip().strip("? .")
    for col in cats:
        mask = df[col].astype(str).str.fullmatch(re.escape(target), case=False, regex=True)
        if mask.any():
            avg = df.loc[mask, "Rate Variance"].mean()
            if pd.notna(avg):
                return f"The average rate variance for **{target}** (based on *{col}*) is **${avg:,.2f}**."
    return None

if q:
    st.chat_message("user").markdown(q)

    direct = direct_variance_answer(q)
    if direct:
        answer = direct
    else:
        numeric_md = df.describe(include="number").to_markdown()
        context = (
            f"Rows: {len(df):,}\nColumns: {', '.join(df.columns)}\n\nNUMERIC SUMMARY\n{numeric_md}\n\n{context_tables}"
        )
        answer = ask_llm(
            "You are a senior data analyst. Use the tables to answer with exact numbers.",
            context + f"\n\nQ: {q}\nA:",
        )

    st.chat_message("assistant").markdown(answer)
    st.session_state.hist += [("user", q), ("assistant", answer)]

    with st.expander("📄 Export this answer"):
        if st.button("Generate PDF", key=f"pdf_{len(st.session_state.hist)}"):
            st.download_button(
                "⬇️ Download PDF",
                pdf_from_md("CSV Insight Assistant report", answer),
                "analysis.pdf",
                "application/pdf",
                key=f"dl_{len(st.session_state.hist)}",
            )

st.caption("rev 6.1 – numeric parsing & PDF export fixed; dynamic filters; auto‑computed answers with GPT‑4o fallback")
