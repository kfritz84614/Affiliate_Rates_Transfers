# app.py – CSV-Insight Assistant 3.1
import os, io, textwrap, smtplib, ssl, datetime
from email.message import EmailMessage

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from ydata_profiling import ProfileReport
from fpdf import FPDF

# ── CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config("CSV Insight Assistant", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in Streamlit secrets"); st.stop()

MODEL  = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

# ── HELPERS ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv(buf: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(buf)

@st.cache_data(show_spinner=False)
def make_profile(df: pd.DataFrame) -> ProfileReport:
    return ProfileReport(df, title="Data profile", minimal=True)

def chat(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def generate_pdf(title: str, body_md: str) -> bytes:
    """Render a one-page PDF; force-wrap long words so FPDF never errors."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    wrapper = textwrap.TextWrapper(width=90, break_long_words=True)
    plain   = body_md.replace("**", "").replace("`", "")
    for line in plain.splitlines():
        for wrapped in wrapper.wrap(line):
            pdf.multi_cell(0, 6, wrapped)

    pdf.set_title(title)
    return bytes(pdf.output())          # fpdf2 returns a bytearray

def send_email(pdf_bytes: bytes, recipient: str):
    if "SMTP_USER" not in st.secrets:
        st.warning("SMTP credentials not set; skipping email.")
        return
    msg       = EmailMessage()
    msg["To"] = recipient
    msg["From"]= st.secrets["SMTP_USER"]
    msg["Subject"] = "CSV Insight Assistant report"
    msg.set_content("Attached is the PDF you requested.")
    msg.add_attachment(pdf_bytes, maintype="application",
                       subtype="pdf", filename="analysis.pdf")

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(st.secrets["SMTP_HOST"], 465, context=ctx) as s:
        s.login(st.secrets["SMTP_USER"], st.secrets["SMTP_PASS"])
        s.send_message(msg)

# ── APP ───────────────────────────────────────────────────────────────────
st.title("📊 CSV Insight Assistant")

uploaded = st.file_uploader("Upload a CSV", type="csv")
if not uploaded:
    st.stop()

df = read_csv(uploaded)

# --- CLEAN CURRENCY COLUMNS ----------------------------------------------
CURRENCY_COLS = ["Rate Variance", "Estimate Amount"]    # extend as needed
for col in CURRENCY_COLS:
    if col in df.columns:
        df[col] = (
            df[col]
              .astype(str)
              .str.replace(r"[^\d.\-]", "", regex=True)
              .replace("", pd.NA)
              .astype(float)
        )
# -------------------------------------------------------------------------

st.success(f"Loaded **{uploaded.name}** – {len(df):,} rows × {len(df.columns)} cols")
st.dataframe(df.head(), use_container_width=True)

# 1️⃣ HEADLINE ANALYSIS ----------------------------------------------------
with st.spinner("Crunching headline stats…"):
    rv   = df["Rate Variance"]
    over = rv[rv > 0]
    save = rv[rv < 0]

    headline = {
        "Total net variance": f"${rv.sum():,.0f}",
        "Jobs with an overrun": f"{len(over):,} ({len(over)/len(df):.1%})",
        "Typical job impact": f"Median ${over.median():,.2f}; 75th pct ${over.quantile(.75):,.2f}",
    }

    # MoM drift
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col:
        df["Month"] = (
            pd.to_datetime(df[date_col], errors="coerce")
              .dt.to_period("M")
              .astype(str)                       # JSON-friendly
        )
        mom = (df.groupby("Month")["Rate Variance"]
                 .mean()
                 .rename("Avg Δ")
                 .to_frame())
        mom_fig = px.line(mom.reset_index().sort_values("Month"),
                          x="Month", y="Avg Δ",
                          title="Average Variance by Month")
    else:
        mom_fig = None

    # Overrun hot-spots
    aff_col = "Affiliate ID"
    hot = (df.groupby(aff_col)
             .agg(jobs=("Rate Variance", "size"),
                  avg_delta=("Rate Variance", "mean"),
                  over_share=("Rate Variance", lambda s: (s>0).mean()))
             .query("jobs >= 10")
             .sort_values("avg_delta", ascending=False)
             .head(10)
             .reset_index())
    hot["avg_delta"]  = hot["avg_delta"].round(2)
    hot["over_share"] = (hot["over_share"]*100).round(1).astype(str) + " %"

    # Consistent savers
    savers = (df.groupby(aff_col)
                .agg(jobs=("Rate Variance", "size"),
                     avg_delta=("Rate Variance", "mean"))
                .query("jobs >= 10 and avg_delta < 0")
                .sort_values("avg_delta")
                .head(5)
                .reset_index())
    savers["avg_delta"] = savers["avg_delta"].round(2)

# Display headline + tables
st.subheader("Headline Findings")
st.table(pd.DataFrame(headline, index=["Value"]).T)

st.subheader("Overrun hot-spots (≥10 jobs)")
st.table(hot)

st.subheader("Consistent savings affiliates")
st.table(savers)

if mom_fig:
    st.subheader("Month-over-month drift")
    st.plotly_chart(mom_fig, use_container_width=True)

# 2️⃣ OPTIONAL PROFILE -----------------------------------------------------
with st.expander("🔍 Detailed ydata-profiling report"):
    if st.button("Generate profile"):
        with st.spinner("Profiling…"):
            pr = make_profile(df)
        st.components.v1.html(pr.to_html(), height=800, scrolling=True)

# 3️⃣ CHAT -----------------------------------------------------------------
st.subheader("💬 Ask follow-up questions")

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

for role, msg in st.session_state.chat_hist:
    st.chat_message(role).markdown(msg)

q = st.chat_input("Ask a question about this dataset")
if q:
    st.chat_message("user").markdown(q)

    # Build compact context
    ctx_tables = {
        "HEADLINE": pd.DataFrame(headline, index=["Value"]).T.to_markdown(),
        "HOT_SPOTS": hot.to_markdown(index=False),
        "SAVERS": savers.to_markdown(index=False),
    }
    if mom_fig:
        ctx_tables["MOM"] = mom.to_markdown()

    context = "\n\n".join(
        f"=== {name} ===\n{tbl}" for name, tbl in ctx_tables.items()
    )

    sys = ("You are a senior data analyst. Use the context tables to answer "
           "with concrete numbers and keep replies under 250 words.")
    ans = chat(sys, context + f"\n\nQUESTION: {q}\nANSWER:")

    st.chat_message("assistant").markdown(ans)
    st.session_state.chat_hist += [("user", q), ("assistant", ans)]

    # PDF download / email
    pdf_bytes = generate_pdf("CSV Insight Assistant report", ans)
    st.download_button("⬇️ Download answer as PDF", pdf_bytes,
                       file_name="analysis.pdf", mime="application/pdf")
    with st.expander("✉️ Email instead"):
        email = st.text_input("Recipient email")
        if st.button("Send PDF"):
            send_email(pdf_bytes, email)
            st.success("Email sent (or skipped if SMTP not configured).")
