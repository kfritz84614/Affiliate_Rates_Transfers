"""Streamlit CSV Insight Assistant – rev 4.1
------------------------------------------------
* Headline panel now shows **Average** _and_ Median variance
* LLM context unchanged – still draws from `df.describe()`
* PDF remains on‑demand via expander
"""
from __future__ import annotations

import os, re, textwrap, ssl
from io import BytesIO
from email.message import EmailMessage

import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF

# ╭──────────────────── CONFIG ────────────────────╮
OPENAI_MODEL = "gpt-4.1"
st.set_page_config("CSV Insight Assistant", page_icon="📊", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
# ╰───────────────────────────────────────────────╯

# ── UTILITIES ────────────────────────────────────

def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _to_numeric(s: pd.Series) -> pd.Series:
    clean = s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(clean, errors="coerce")

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("date", "time")):
            df[c] = _to_datetime(df[c])
        elif any(k in lc for k in ("variance", "amount", "cost", "price")):
            df[c] = _to_numeric(df[c])
    return df

def wrap_long(text: str, width: int = 90):
    for line in text.splitlines():
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

def send_email(pdf: bytes, recipient: str):
    if not {"SMTP_HOST", "SMTP_USER", "SMTP_PASS"}.issubset(st.secrets):
        st.warning("SMTP creds not configured."); return
    import smtplib
    msg = EmailMessage(); msg["To"] = recipient; msg["From"] = st.secrets["SMTP_USER"]
    msg["Subject"] = "CSV Insight Assistant report"; msg.set_content("Attached PDF.")
    msg.add_attachment(pdf, maintype="application", subtype="pdf", filename="analysis.pdf")
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(st.secrets["SMTP_HOST"], 465, context=ctx) as s:
        s.login(st.secrets["SMTP_USER"], st.secrets["SMTP_PASS"]); s.send_message(msg)

# LLM helper

def ask_llm(system: str, user: str) -> str:
    if not OPENAI_API_KEY:
        return "*(Add OPENAI_API_KEY to get LLM answers)*"
    import openai; openai.api_key = OPENAI_API_KEY
    resp = openai.chat.completions.create(model=OPENAI_MODEL, messages=[
        {"role":"system","content":system},
        {"role":"user","content":user}
    ], temperature=0.2)
    return resp.choices[0].message.content.strip()

# ── APP ───────────────────────────────────────────

st.title("📊 CSV Insight Assistant")
file = st.file_uploader("Upload CSV", type="csv")
if not file: st.stop()

df = clean_df(pd.read_csv(file))
st.dataframe(df.head(), use_container_width=True)

# ── HEADLINE METRICS ─────────────────────────────
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

# ── CHAT ----------------------------------------------------------------
if "hist" not in st.session_state: st.session_state.hist = []
for role, msg in st.session_state.hist:
    st.chat_message(role).markdown(msg)

question = st.chat_input("Ask a question about the dataset…")
if question:
    st.chat_message("user").markdown(question)
    numeric_md = df.describe(include="number").to_markdown()
    context = f"Rows: {len(df):,}\nColumns: {', '.join(df.columns)}\n\nNUMERIC SUMMARY\n{numeric_md}"
    answer = ask_llm("You are a senior data analyst. Answer with specific numbers.", context + f"\n\nQ: {question}\nA:")
    st.chat_message("assistant").markdown(answer)
    st.session_state.hist += [("user", question), ("assistant", answer)]

    with st.expander("📄 Export this answer"):
        if st.button("Generate PDF", key=f"pdf_{len(st.session_state.hist)}"):
            pdf_bytes = pdf_from_md("CSV Insight Assistant report", answer)
            st.download_button("⬇️ Download PDF", pdf_bytes, "analysis.pdf", "application/pdf", key=f"dl_{len(st.session_state.hist)}")
            to = st.text_input("Email to…", key=f"email_{len(st.session_state.hist)}")
            if st.button("Send", key=f"send_{len(st.session_state.hist)}"):
                send_email(pdf_bytes, to)
                st.success("Sent (if SMTP set).")

# ── OPTIONAL PROFILE ─────────────────────────────
with st.expander("🔍 Detailed profile (ydata-profiling)"):
    if st.button("Generate profile"):
        from ydata_profiling import ProfileReport
        pr = ProfileReport(df, title="Profile", minimal=True)
        st.components.v1.html(pr.to_html(), height=800, scrolling=True)

st.caption("v4.1 – avg variance added • LLM requires API key")
