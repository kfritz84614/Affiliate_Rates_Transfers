# app.py — CSV Insight Assistant
# -----------------------------------------------
# • Upload a CSV and instantly explore it
# • One-click profiling (ydata-profiling)
# • Quick trend chart wizard (Plotly)
# • Natural-language Q&A powered by OpenAI
# -----------------------------------------------

import os, textwrap, io
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from ydata_profiling import ProfileReport

# ----------  Streamlit page setup  ----------
st.set_page_config(page_title="CSV Insight Assistant", layout="wide")
st.title("📊 CSV Insight Assistant")

# ----------  OpenAI client  ----------
OPENAI_API_KEY = (
    st.secrets["OPENAI_API_KEY"]        # Streamlit Cloud secrets
    if "OPENAI_API_KEY" in st.secrets   # (fallback to env for local dev)
    else os.getenv("OPENAI_API_KEY")
)
if not OPENAI_API_KEY:
    st.error("❌ No OpenAI key found in Streamlit secrets or $OPENAI_API_KEY.")
    st.stop()

openai = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"   # good balance of speed / context

# ----------  Helpers  ----------
@st.cache_data(show_spinner=False)
def load_csv(buffer: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(buffer)

@st.cache_data(show_spinner=False)
def profile(df: pd.DataFrame) -> ProfileReport:
    return ProfileReport(df, title="Data Profiling Report", minimal=True)

def ask_llm(df: pd.DataFrame, question: str) -> str:
    numeric_md = df.describe(include="number").to_markdown()
    prompt = textwrap.dedent(
        f"""
        You are a senior data analyst.
        The user uploaded a CSV with {len(df):,} rows.
        Columns: {', '.join(df.columns)}.

        **Numeric summary (markdown)**:
        {numeric_md}

        Answer the question **succinctly** and suggest one or two follow-up analyses.

        Question: {question}
        """
    )
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ----------  File upload  ----------
file = st.file_uploader("📥  Upload a CSV file", type="csv")
if not file:
    st.info("Drag a CSV above to begin.")
    st.stop()

df = load_csv(file)
st.success(f"Loaded **{file.name}** – {len(df):,} rows × {len(df.columns)} cols")
st.dataframe(df.head(25), use_container_width=True)

# ----------  Quick EDA / profiling  ----------
with st.expander("🔍  One-click Profiling"):
    if st.button("Generate profiling report"):
        with st.spinner("Running ydata-profiling… (may take a minute)"):
            pr = profile(df)
        st.components.v1.html(pr.to_html(), height=800, scrolling=True)

# ----------  Trend chart wizard  ----------
num_cols = df.select_dtypes("number").columns
if len(num_cols) >= 2:
    st.subheader("📈  Quick Trend Plot")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", num_cols, key="x_axis")
    with col2:
        y_axis = st.selectbox("Y-axis", num_cols, key="y_axis")
    if x_axis and y_axis:
        chart = px.line(
            df.sort_values(by=x_axis),
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} over {x_axis}",
        )
        st.plotly_chart(chart, use_container_width=True)

# ----------  Ask AI  ----------
st.subheader("🤖  Ask a question about your data")
query = st.text_area("Type your question here…", placeholder="e.g. Which products are growing fastest?")
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking…"):
            answer = ask_llm(df, query)
        st.markdown(answer)
        st.download_button(
            "💾  Download answer (.md)",
            data=answer.encode(),
            file_name="analysis.md",
            mime="text/markdown",
        )
