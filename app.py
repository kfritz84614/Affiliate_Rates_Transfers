"""Streamlit CSV Insight Assistant – rev 5.0
------------------------------------------------------------------
Dynamic context:
• Builds per‑category **mean Rate Variance** tables for *all* categorical
  columns (≤ 120 unique values) – Affiliate ID, Chauffeur, City …
• When the user asks a question the assistant now receives *all* those
  tables → it can answer “average variance for Affiliate 972”, “for SUV”,
  “for Chicago”, etc. without extra prompting.
• Headline panel unchanged; PDF still on‑demand.
"""
from __future__ import annotations

import os, re, textwrap, ssl
from io import BytesIO
from email.message import EmailMessage

import pandas as pd
import streamlit as st
from fpdf import FPDF

# ╭──────────────── CONFIG ───────────────╮
OPENAI_MODEL = "gpt-4o-mini"
st.set_page_config("CSV Insight Assistant", page_icon="📊", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
# ╰───────────────────────────────────────╯

# ── CLEANERS ────────────────────────────

def _to_datetime(s: pd.Series) -> pd.Series: return pd.to_datetime(s, errors="coerce")

def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("date", "time")):
            df[c] = _to_datetime(df[c])
        elif any(k in lc for k in ("variance", "amount", "cost", "price")):
            df[c] = _to_numeric(df[c])
    return df

# ── PDF ─────────────────────────────────

def wrap_long(txt: str, width: int = 90):
    for line in txt.splitlines():
        while line:
            yield line[:width]
            line = line[width:]

def pdf_from_md(title: str, md: str) -> bytes:
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page(); pdf.set_font("Helvetica", size=12)
    plain = re.sub(r"[*`_]", "", md)
    for seg in wrap_long(plain, 90):
        pdf.multi_cell(0, 6, seg)
    pdf.set_title(title)
    buf = BytesIO(); pdf.output(buf); return buf.getvalue()

# ── LLM helper ──────────────────────────

def ask_llm(system: str, user: str) -> str:
    if not OPENAI_API_KEY:
        return "*(Set OPENAI_API_KEY to enable LLM answers)*"
    import openai; openai.api_key = OPENAI_API_KEY
    resp = openai.chat.completions.create(model=OPENAI_MODEL, temperature=0.2,
        messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return resp.choices[0].message.content.strip()

# ── APP ────────────────────────────────

st.title("📊 CSV Insight Assistant")
file = st.file_uploader("Upload CSV", type="csv")
if not file: st.stop()

df = clean_df(pd.read_csv(file))
st.dataframe(df.head(), use_container_width=True)

# ── Headline KPIs ───────────────────────
if "Rate Variance" in df.columns:
    rv = df["Rate Variance"].dropna()
    over = rv[rv > 0]
    headline = {
        "Total net variance": f"${rv.sum():,.0f}",
        "Average variance (mean)": f"${rv.mean():,.2f}",
        "Median variance": f"${rv.median():,.2f}",
        "Jobs with an overrun": f"{len(over):,} ({len(over)/len(df):.1%})",
    }
    st.subheader("📌 Headline summary")
    st.table(pd.DataFrame(headline, index=["Value"]))

# ── Build per‑category variance tables ──
CAT_LIMIT = 120   # skip very high‑cardinality cols
cat_tables: list[str] = []
for col in df.select_dtypes("object").columns:
    if df[col].nunique() <= CAT_LIMIT and "rate variance" in df.columns:
        tbl = (df[[col, "Rate Variance"]]
               .dropna()
               .groupby(col)["Rate Variance"].mean()
               .round(2)
               .sort_values(ascending=False))
        if not tbl.empty:
            cat_tables.append(f"=== {col.upper()} — Avg Rate Variance ===\n{tbl.to_markdown()}")
            # preview first 10 rows
            with st.expander(f"🔎 {col} averages (top 10)"):
                st.table(tbl.head(10))

context_cats = "\n\n".join(cat_tables)

# ── Chat ────────────────────────────────
if "hist" not in st.session_state: st.session_state.hist = []
for role, msg in st.session_state.hist: st.chat_message(role).markdown(msg)

q = st.chat_input("Ask a question … e.g. average variance for Affiliate 972")
if q:
    st.chat_message("user").markdown(q)

    numeric_md = df.describe(include="number").to_markdown()
    ctx = f"Rows: {len(df):,}\nColumns: {', '.join(df.columns)}\n\nNUMERIC SUMMARY\n{numeric_md}\n\n{context_cats}"

    sys = "You are a senior data analyst. Use the tables to answer with exact numbers."
    ans = ask_llm(sys, ctx + f"\n\nQ: {q}\nA:")
    st.chat_message("assistant").markdown(ans)
    st.session_state.hist += [("user", q), ("assistant", ans)]

    with st.expander("📄 Export this answer"):
        if st.button("Generate PDF", key=f"pdf_{len(st.session_state.hist)}"):
            st.download_button("⬇️ Download PDF", pdf_from_md("CSV Insight Assistant report", ans),
                               "analysis.pdf", "application/pdf", key=f"dl_{len(st.session_state.hist)}")

st.caption("v5.0 – dynamic per‑category tables feed the LLM for any city / affiliate / chauffeur question")
