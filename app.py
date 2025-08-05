# app.py ― CSV-Insight Assistant 2.1
import os, io, textwrap, smtplib, ssl
from email.message import EmailMessage

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from ydata_profiling import ProfileReport
from fpdf import FPDF

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

def chat_completion(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def generate_pdf(title: str, body_md: str) -> bytes:
    """Render a one-page PDF from plain text (wrap long lines)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    plain = body_md.replace("**", "").replace("`", "")
    for line in plain.splitlines():
        for wrapped in textwrap.wrap(line, width=90):
            pdf.multi_cell(0, 6, wrapped)

    pdf.set_title(title)
    return bytes(pdf.output())             # fpdf2 already returns a bytearray
                                   # fpdf2 returns bytes

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

# ---------- UI ----------
st.title("📊 CSV Insight Assistant")

uploaded = st.file_uploader("Upload a CSV", type="csv")
if not uploaded:                          # (unchanged)
    st.stop()

df = read_csv(uploaded)

# ✱ NEW — force numeric on any currency-looking column -----------------
currency_cols = ["Rate Variance"]          # add others if needed
for col in currency_cols:
    if col in df.columns:
        df[col] = (
            df[col]
              .astype(str)                 # be safe
              .str.replace(r"[^\d.\-]", "", regex=True)  # strip $ and commas
              .replace("", pd.NA)          # empty → NaN
              .astype(float)
        )

# ----------------------------------------------------------------------
st.success(f"Loaded **{uploaded.name}** – {len(df):,} rows × {len(df.columns)} cols")
st.dataframe(df.head(), use_container_width=True)

# ---- optional profiling
with st.expander("🔍 One-click profiling"):
    if st.button("Generate profile"):
        with st.spinner("Profiling…"):
            pr = make_profile(df)
        st.components.v1.html(pr.to_html(), height=800, scrolling=True)

# ---- quick chart
numeric_cols = df.select_dtypes("number").columns
if len(numeric_cols) >= 2:
    st.subheader("📈 Quick trend plot")
    c1, c2 = st.columns(2)
    x = c1.selectbox("X-axis", numeric_cols, key="x_axis")
    y = c2.selectbox("Y-axis", numeric_cols, key="y_axis")
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

    # -------- build rich LLM context --------
    numeric_md = df.describe(include="number").to_markdown()

    # adjust these column names if your CSV differs
    city_col = "Pickup City"
    var_col  = "Rate Variance"
    if city_col in df.columns and var_col in df.columns:
        top_cities_md = (
            df.groupby(city_col)[var_col]
              .mean()
              .sort_values(ascending=False)
              .head(15)
              .to_markdown()
        )
    else:
        top_cities_md = "*[Columns not present]*"

    context = f"""
DATA SNAPSHOT
Rows: {len(df):,}
Columns: {', '.join(df.columns)}

=== NUMERIC SUMMARY ===
{numeric_md}

=== TOP 15 {city_col.upper()} BY AVG {var_col.upper()} ===
{top_cities_md}
"""

    system = (
        "You are a senior data analyst. Use the context below to answer with "
        "specific numbers directly from the data. Be concise but thorough."
    )
    user = context + f"\n\nQUESTION: {question}\nANSWER:"

    with st.spinner("Analyzing…"):
        answer = chat_completion(system, user)

    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history += [("user", question), ("assistant", answer)]

    # -------- PDF download & email --------
    pdf_bytes = generate_pdf("CSV Insight Assistant report", answer)
    st.download_button("⬇️ Download answer as PDF", pdf_bytes,
                       file_name="analysis.pdf", mime="application/pdf")
    with st.expander("✉️ Email instead"):
        email_to = st.text_input("Recipient email")
        if st.button("Send PDF"):
            send_email(pdf_bytes, email_to)
            st.success("Email sent (or skipped if SMTP not configured).")
