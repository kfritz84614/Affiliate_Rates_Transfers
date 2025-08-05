# app.py ― CSV-Insight Assistant 2.0
import os, io, textwrap, smtplib, ssl, base64
from email.message import EmailMessage

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from ydata_profiling import ProfileReport
from fpdf import FPDF                       # lightweight PDF helper

# ---------- config ----------
st.set_page_config("CSV Insight Assistant", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in Streamlit Secrets"); st.stop()
MODEL = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def read_csv(buf: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(buf)

@st.cache_data(show_spinner=False)
def make_profile(df: pd.DataFrame) -> ProfileReport:
    return ProfileReport(df, title="Data profile", minimal=True)

def chat_completion(system_prompt, user_prompt):
    msg = [{"role": "system", "content": system_prompt},
           {"role": "user", "content": user_prompt}]
    return client.chat.completions.create(model=MODEL, messages=msg,
                                          temperature=0.2).choices[0].message.content.strip()

def generate_pdf(title: str, body_md: str) -> bytes:
    """Render simple PDF from markdown string (one page)."""
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page(); pdf.set_font("Helvetica", size=12)
    for line in body_md.splitlines():
        pdf.multi_cell(0, 6, line)
    pdf.set_title(title)
    return pdf.output(dest="S").encode()

def send_email(pdf_bytes: bytes, recipient: str):
    if "SMTP_USER" not in st.secrets:        # abort silently for demo
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

# ---------- UI ----------
st.title("📊 CSV Insight Assistant")

uploaded = st.file_uploader("Upload a CSV", type="csv")
if not uploaded: st.stop()

df = read_csv(uploaded)
st.success(f"Loaded **{uploaded.name}** – {len(df):,} rows × {len(df.columns)} cols")
st.dataframe(df.head(), use_container_width=True)

# ---- optional profiling
with st.expander("🔍 One-click profiling"):
    if st.button("Generate profile"):
        with st.spinner("Profiling…"): pr = make_profile(df)
        st.components.v1.html(pr.to_html(), height=800, scrolling=True)

# ---- quick chart
num_cols = df.select_dtypes("number").columns
if len(num_cols) >= 2:
    st.subheader("📈 Quick trend plot")
    c1, c2 = st.columns(2)
    x = c1.selectbox("X-axis", num_cols)
    y = c2.selectbox("Y-axis", num_cols)
    if x and y:
        fig = px.line(df.sort_values(x), x=x, y=y, title=f"{y} over {x}")
        st.plotly_chart(fig, use_container_width=True)

# ---------- conversational analysis ----------
st.subheader("💬 Chat about your data")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    st.chat_message(role).markdown(content)

question = st.chat_input("Ask a question")
if question:
    st.chat_message("user").markdown(question)
    with st.spinner("Analyzing…"):
        numeric_md = df.describe(include="number").to_markdown()
        system = "You are a senior data analyst. Use the data summary to answer precisely."
        user   = textwrap.dedent(f"""
            CSV rows: {len(df):,}
            Columns: {', '.join(df.columns)}
            Numeric summary:\n{numeric_md}

            Question: {question}
            """)
        answer = chat_completion(system, user)
    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("assistant", answer))

    # ---- PDF + email options
    pdf_bytes = generate_pdf("CSV Insight Assistant report", answer)
    st.download_button("⬇️ Download answer as PDF", pdf_bytes,
                       file_name="analysis.pdf", mime="application/pdf")
    with st.expander("✉️ Email instead"):
        email_to = st.text_input("Recipient email")
        if st.button("Send PDF"):
            send_email(pdf_bytes, email_to)
            st.success("Email sent (or skipped if SMTP not configured).")
