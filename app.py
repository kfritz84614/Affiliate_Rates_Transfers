"""Streamlit CSV Insight Assistant – re‑worked
------------------------------------------------
Goals
-----
* Robust data cleaning → convert dates, currency, numeric
* Conversational Q&A powered by OpenAI (or your LLM of choice)
* Optional one‑click PDF summary instead of auto‑generated each turn
* Simpler, safer visualisations (no Period objects, no NaNs)
* Keeps IDs as labels (no text split), treats locations as single categorical values
"""
from __future__ import annotations

import re
import textwrap
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF

# 👉───────────── CONFIGURATION ───────────────────────────────────────────
st.set_page_config("CSV Insight Assistant", page_icon="💬", layout="centered")
OPENAI_MODEL = "gpt-3.5-turbo"  # or change to gpt‑4o etc.

# 👉───────────── HELPER FUNCTIONS ────────────────────────────────────────

def _to_datetime(col: pd.Series) -> pd.Series:
    """Convert mixed date strings to datetime without raising."""
    return pd.to_datetime(col, errors="coerce", dayfirst=False, yearfirst=False)


def _to_numeric(col: pd.Series) -> pd.Series:
    """Turn $1,234.56 or 1,234 into float – keeps NaN where fails."""
    cleaned = col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Infer & coerce common column types (dates, currency, numbers)."""
    date_keywords = {"date", "time"}
    currency_keywords = {"cost", "price", "variance", "amount"}
    num_like = {"rating", "id", "jobs", "qty", "count"}

    for col in df.columns:
        lowered = col.lower()
        if any(k in lowered for k in date_keywords):
            df[col] = _to_datetime(df[col])
        elif any(k in lowered for k in currency_keywords):
            df[col] = _to_numeric(df[col])
        elif any(k in lowered for k in num_like):
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def split_long_tokens(txt: str, limit: int = 40) -> list[str]:
    """FPDF cannot render an *un‑breakable* 100+ char token – add soft breaks."""
    out: list[str] = []
    for token in re.split(r"(\s+)", txt):
        if len(token) > limit and not token.isspace():
            out.extend(re.findall(rf".{{1,{limit}}}", token))
        else:
            out.append(token)
    return out


def generate_pdf(title: str, body_md: str) -> bytes:
    """Return ready‑to‑download PDF bytes from Markdown *body_md*."""
    wrapper = textwrap.TextWrapper(width=90, break_long_words=False, break_on_hyphens=False)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    plain = re.sub(r"[*_`]", "", body_md)  # strip markdown emphasises
    for raw_line in plain.splitlines():
        # apply soft‑wrap first
        for wrapped_line in wrapper.wrap(raw_line) or [""]:
            for segment in split_long_tokens(wrapped_line):
                pdf.multi_cell(0, 6, segment)
    pdf.set_title(title)

    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


# 👉───────────── LAYOUT & SESSION STATE ──────────────────────────────────

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist: list[tuple[str, str]] = []

st.title("💬 CSV Insight Assistant")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])  # simple – extend as you wish
if not uploaded:
    st.info("Upload a CSV to start ↗️")
    st.stop()

# ---------- DATA PREP ----------------------------------------------------

df = pd.read_csv(uploaded)
df = clean_dataframe(df)

# quick preview
with st.expander("📄 Preview (first 500 rows)"):
    st.dataframe(df.head(500))

# ---------- CHAT ---------------------------------------------------------
question = st.chat_input("Ask a question about the data …")

# replay history
for speaker, msg in st.session_state.chat_hist:
    with st.chat_message(speaker):
        st.markdown(msg)

if question:
    st.chat_message("user").markdown(question)

    with st.spinner("Thinking hard about your data …"):
        # summarise numeric columns for the LLM context
        numeric_md = df.describe(include="number").to_markdown()
        system = (
            "You are a senior data analyst. Provide concise, insight‑driven answers. "
            "Return well‑structured Markdown with bullet points and short tables where helpful."
        )
        user_prompt = f"Dataset rows: {len(df):,}\n\nNumeric summary:\n{numeric_md}\n\nUser question: {question}"

        try:
            import openai

            if "OPENAI_API_KEY" in st.secrets:
                openai.api_key = st.secrets["OPENAI_API_KEY"]
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:  # fall‑back stub for offline / dev
            answer = f"*(LLM call failed – {e})*\n\n**Stub answer**: There are {len(df):,} rows. Please configure your OpenAI key."

    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_hist += [("user", question), ("assistant", answer)]

    # --------- OPTIONAL PDF (on demand) ----------------------------------
    with st.expander("📄 Export" ):
        if st.button("Generate PDF summary of this answer"):
            pdf_bytes = generate_pdf("CSV Insight Assistant report", answer)
            st.download_button("⬇️ Download PDF", pdf_bytes, "analysis.pdf", "application/pdf")

# ---------- QUICK INSIGHT SECTIONS --------------------------------------

# 1️⃣ Month‑over‑month drift chart (if Month column exists & numeric var too)
if {"Month", "Rate Variance"}.issubset(df.columns):
    try:
        mom_df = (
            df.dropna(subset=["Month", "Rate Variance"])
            .assign(Month=lambda d: d["Month"].astype(str))
            .groupby("Month", as_index=False)["Rate Variance"]
            .mean()
            .sort_values("Month")
        )
        if not mom_df.empty:
            st.subheader("📈 Month‑over‑month average variance")
            fig = px.line(mom_df, x="Month", y="Rate Variance", markers=True)
            fig.update_layout(yaxis_title="Avg $ variance")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as err:
        st.warning(f"Couldn't draw MoM chart: {err}")

# 2️⃣ ydata‑profiling full report (heavy – optional)
with st.expander("🔍 Detailed profiling report"):
    if st.button("Generate profile (may take 1‑2 min)"):
        from ydata_profiling import ProfileReport

        pr = ProfileReport(df, title="Data profile", minimal=True)
        st.components.v1.html(pr.to_html(), height=1000, scrolling=True)

# 3️⃣ Simple top‑N overrun cities (as requested) – robust to strings
city_col = next((c for c in df.columns if "city" in c.lower() and "pickup" in c.lower()), None)
var_col = "Rate Variance" if "Rate Variance" in df.columns else None

if city_col and var_col:
    try:
        top_cities = (
            df[[city_col, var_col]].dropna()
            .groupby(city_col, as_index=False)[var_col]
            .mean()
            .sort_values(var_col, ascending=False)
            .head(10)
        )
        st.subheader("🏙️ Highest average variance cities (top 10)")
        st.table(top_cities)
    except Exception as err:
        st.warning(f"City breakdown unavailable: {err}")

# ---------- FOOTER -------------------------------------------------------

st.caption("Made with ❤️ for data‑driven teams.  |  v0.4‑revamped")
