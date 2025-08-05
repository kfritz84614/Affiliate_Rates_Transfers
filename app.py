"""
CSV-Insight Assistant – rev 3.2
• fixes “FPDFException: Not enough horizontal space…”
• forces proper dtypes (IDs→string, dates→datetime, $→float)
• trims ydata-profiling to the essentials
"""
import os, io, textwrap, smtplib, ssl, re
from email.message import EmailMessage

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from ydata_profiling import ProfileReport
from fpdf import FPDF, FPDFException        # ← catch PDF overflow

# ─── CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config("CSV Insight Assistant", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in Streamlit secrets"); st.stop()

MODEL  = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

# ─── HELPERS ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv(buf: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(buf)

def force_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """IDs→str, $→float, dates→datetime, locations→str (single token)."""
    for c in df.columns:
        cl  = c.lower()
        ser = df[c]

        # currency / numeric with symbols
        if any(k in cl for k in ("amount", "cost", "price", "variance", "rate", "$")):
            df[c] = pd.to_numeric(
                ser.astype(str).str.replace(r"[^\d.\-]", "", regex=True),
                errors="coerce"
            )
            continue

        # dates / times
        if any(k in cl for k in ("date", "time", "timestamp")):
            df[c] = pd.to_datetime(ser, errors="coerce")
            continue

        # identifiers
        if "id" in cl:
            df[c] = ser.astype(str)
            continue

        # locations → keep as string, no tokenisation
        if any(k in cl for k in ("city", "state", "market", "country", "location")):
            df[c] = ser.astype(str)

    return df

@st.cache_data(show_spinner=False)
def make_profile(df: pd.DataFrame) -> ProfileReport:
    cfg = {
        "correlations": {"pearson": False, "spearman": False, "kendall": False,
                         "phi_k": False, "cramers": False, "recoded": False},
        "missing_diagrams": {"heatmap": False, "dendrogram": False},
        "samples": {"head": 0, "tail": 0},
    }
    return ProfileReport(df, title="Data profile", minimal=True, config=cfg)

def chat(sys: str, usr: str) -> str:
    m = [{"role":"system","content":sys},{"role":"user","content":usr}]
    return client.chat.completions.create(model=MODEL, messages=m,
                                          temperature=0.2).choices[0].message.content.strip()

def _safe_lines(text: str, width: int = 90):
    """Yield chunks ≤ width—even if there are no spaces."""
    for line in text.splitlines():
        while line:
            yield line[:width]
            line = line[width:]

def generate_pdf(title: str, body_md: str) -> bytes:
    """PDF builder immune to long strings."""
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page(); pdf.set_font("Helvetica", size=12)

    plain = re.sub(r"[*`_]", "", body_md)  # strip markdown markup
    for seg in _safe_lines(plain, 90):
        try:
            pdf.multi_cell(0, 6, seg)
        except FPDFException:              # still too wide → hard-split
            for chunk in _safe_lines(seg, 60):
                pdf.multi_cell(0, 6, chunk)

    pdf.set_title(title)
    return bytes(pdf.output())

def send_email(pdf_bytes: bytes, recipient: str):
    if "SMTP_USER" not in st.secrets:
        st.warning("SMTP not configured.")
        return
    msg = EmailMessage()
    msg["To"], msg["From"] = recipient, st.secrets["SMTP_USER"]
    msg["Subject"] = "CSV Insight Assistant report"
    msg.set_content("Attached is the PDF you requested.")
    msg.add_attachment(pdf_bytes, maintype="application",
                       subtype="pdf", filename="analysis.pdf")
    with smtplib.SMTP_SSL(st.secrets["SMTP_HOST"], 465, context=ssl.create_default_context()) as s:
        s.login(st.secrets["SMTP_USER"], st.secrets["SMTP_PASS"])
        s.send_message(msg)

# ─── APP ─────────────────────────────────────────────────────────────────
st.title("📊 CSV Insight Assistant")

buf = st.file_uploader("Upload a CSV", type="csv")
if not buf:
    st.stop()

df = force_dtypes(read_csv(buf))
st.success(f"Loaded **{buf.name}** — {len(df):,} rows × {len(df.columns)} cols")
st.dataframe(df.head(), use_container_width=True)

# ── quick profile (trimmed) ─────────────────────────────────────────────
with st.expander("🔍 Lightweight profile"):
    if st.button("Generate profile"):
        with st.spinner("Profiling…"):
            pr = make_profile(df)
        st.components.v1.html(pr.to_html(), height=600, scrolling=True)

# ── headline KPIs ───────────────────────────────────────────────────────
rv = df["Rate Variance"]
over = rv[rv > 0]
headline = {
    "Total net variance": f"${rv.sum():,.0f}",
    "Jobs with an overrun": f"{len(over):,} ({len(over)/len(df):.1%})",
    "Median overrun": f"${over.median():,.2f}",
}
st.subheader("Headline")
st.table(pd.DataFrame(headline, index=["Value"]).T)

# ── chat interface ──────────────────────────────────────────────────────
st.subheader("💬 Ask questions")
if "hist" not in st.session_state:
    st.session_state.hist = []

for role, msg in st.session_state.hist:
    st.chat_message(role).markdown(msg)

q = st.chat_input("Ask a question")
if q:
    st.chat_message("user").markdown(q)

    ctx = f"ROWS={len(df):,}\nCOLUMNS={', '.join(df.columns)}\n" \
          f"NUMERIC SUMMARY\n{df.describe(include='number').to_markdown()}"
    ans = chat("You are a senior data analyst. Answer with data-backed numbers.",
               ctx + f"\n\nQUESTION: {q}\nANSWER:")

    st.chat_message("assistant").markdown(ans)
    st.session_state.hist += [("user", q), ("assistant", ans)]

    # PDF / email
    pdf_bytes = generate_pdf("CSV Insight Assistant report", ans)
    st.download_button("⬇️ PDF", pdf_bytes, "analysis.pdf", "application/pdf")
    with st.expander("✉️ Email"):
        to = st.text_input("Recipient email")
        if st.button("Send"):
            send_email(pdf_bytes, to)
            st.success("Sent (or skipped if SMTP absent).")
