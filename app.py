"""
Streamlit CSV Explorer – v9.0 (2025‑08‑06)
------------------------------------------------
• Upload any CSV ➜ automatic cleaning & type coercion
• GPT‑4o‑mini function‑calling for questions & chart suggestions
• Auto‑render up to 5 LLM‑picked Plotly charts on load
• PDF export, cost logging, and rock‑solid error handling
"""

from __future__ import annotations

import json
import textwrap
import time
from io import BytesIO
from typing import Any, Dict, List, Tuple

import openai
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF

# ── Config & global objects ──────────────────────────────────────────────

st.set_page_config(
    page_title="CSV Explorer AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

openai.api_key = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"

# cost tracker (rough calc – update if model pricing changes)
TOKEN_PRICE_USD = 0.00001  # ≈ $0.01 per 1k tokens → adjust as needed
if "_cost_usd" not in st.session_state:
    st.session_state._cost_usd = 0.0

def _add_cost(token_count: int) -> None:
    st.session_state._cost_usd += token_count * TOKEN_PRICE_USD / 1000

# ── Helpers: data cleaning & caching ─────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    """Read CSV to DataFrame and coerce numerics / dates."""
    df = pd.read_csv(file)

    # Numeric coercion – only columns that *look* numeric
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        sample = df[col].head(20).astype(str).str.replace(r"[\s,$%()]", "", regex=True)
        pct_numericish = (sample.str.match(r"^-?\d*[\.,]?\d+$").mean())
        if pct_numericish > 0.8:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[,$%]", "", regex=True)
                .str.replace("(", "-", regex=False)
                .str.replace(")", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Datetime coercion
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass

    return df

# ── LLM function helpers exposed to GPT ──────────────────────────────────

def aggregate(by: str, target: str | None, metric: str = "sum", top_n: int | None = None) -> pd.DataFrame:  # noqa: D401
    """Group *df* (from session) by *by* column and compute *metric* on *target*.

    metric: sum | mean | count
    """
    df = st.session_state.df
    if target is None:
        target = df.columns[0]
    if metric == "count":
        ser = df.groupby(by)[target].count()
    elif metric == "mean":
        ser = df.groupby(by)[target].mean()
    else:
        ser = df.groupby(by)[target].sum()
    out = ser.reset_index().rename(columns={target: metric})
    if top_n:
        out = out.nlargest(top_n, metric)
    return out

def get_rows(where: str, columns: List[str] | None = None, limit: int = 20) -> pd.DataFrame:
    """Return rows matching a pandas-query style *where* expression."""
    df = st.session_state.df
    try:
        subset = df.query(where)
    except Exception as e:
        st.warning(f"❌ Bad query: {e}")
        return pd.DataFrame()
    if columns:
        subset = subset[columns]
    return subset.head(limit)

FUNCTIONS_SPEC = [
    {
        "name": "aggregate",
        "description": "Grouped statistics of the dataset",
        "parameters": {
            "type": "object",
            "properties": {
                "by": {"type": "string"},
                "target": {"type": ["string", "null"]},
                "metric": {"type": "string", "enum": ["sum", "mean", "count"]},
                "top_n": {"type": ["integer", "null"]},
            },
            "required": ["by"],
        },
    },
    {
        "name": "get_rows",
        "description": "Filter raw rows by expression",
        "parameters": {
            "type": "object",
            "properties": {
                "where": {"type": "string"},
                "columns": {"type": ["array", "null"], "items": {"type": "string"}},
                "limit": {"type": "integer"},
            },
            "required": ["where"],
        },
    },
]

# ── LLM helpers ──────────────────────────────────────────────────────────

def call_llm(messages: List[Dict[str, Any]], functions: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Wrapper with retry & cost tracking."""
    for attempt in range(5):
        try:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                functions=functions,
            )
            _add_cost(response.usage.total_tokens)  # type: ignore[attr-defined]
            return response
        except openai.RateLimitError:
            delay = 2 ** attempt
            time.sleep(delay)
    raise RuntimeError("OpenAI API failed after retries")

# ── UI: Sidebar upload & cleaning ───────────────────────────────────────

with st.sidebar:
    st.markdown("## 📤 Upload CSV")
    uploaded = st.file_uploader("Choose file", type=["csv"])
    if uploaded:
        st.session_state.df = load_csv(uploaded)
        st.success("File loaded & cleaned!")
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are a data analyst. Use helper functions when appropriate."
            }
        ]
        # reset charts when new file uploaded
        st.session_state.pop("_charts_rendered", None)
        st.session_state.pop("_chart_specs", None)

    # show cost so far
    st.markdown(f"**Usage cost:** ${st.session_state._cost_usd:.4f}")

# ── Main layout blocks ──────────────────────────────────────────────────

def show_headline_kpis(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    numeric_cols = df.select_dtypes("number").columns
    if not numeric_cols.empty:
        col2.metric("Σ of first numeric col", f"{df[numeric_cols[0]].sum():,.0f}")
        overruns = numeric_cols.intersection([c for c in df.columns if "overrun" in c.lower()])
        if not overruns.empty:
            share = (df[overruns[0]] > 0).mean() * 100
            col3.metric("Over‑run share", f"{share:.1f}%")


def render_chart(spec: Dict[str, str]):
    """Create Plotly chart from a minimal spec."""
    chart_type = spec.get("type", "bar")
    x = spec.get("x")
    y = spec.get("y")
    agg = spec.get("agg", "sum")
    df = st.session_state.df
    if not (x and y):
        st.warning("Spec missing x or y – skipping chart")
        return
    if agg == "count":
        df_plot = df.groupby(x)[y].count().reset_index(name="count")
        y = "count"
    else:
        df_plot = df.groupby(x)[y].agg(agg).reset_index()
    if chart_type == "line":
        fig = px.line(df_plot, x=x, y=y)
    elif chart_type == "scatter":
        fig = px.scatter(df_plot, x=x, y=y)
    elif chart_type == "heatmap":
        fig = px.density_heatmap(df, x=x, y=y)
    else:
        fig = px.bar(df_plot, x=x, y=y)
    st.plotly_chart(fig, use_container_width=True)


if df := st.session_state.get("df"):
    show_headline_kpis(df)

    # ── LLM‑recommended charts on first load ────────────────
    if "_charts_rendered" not in st.session_state:
        user_prompt = "Provide up to 5 useful chart specs (json) for this dataset. Return a JSON list."
        resp = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            functions=None,
        )
        try:
            charts = json.loads(resp.choices[0].message.content)
            st.session_state._chart_specs = charts[:5]
        except Exception:
            st.session_state._chart_specs = []
        st.session_state._charts_rendered = True

    for spec in st.session_state.get("_chart_specs", []):
        try:
            render_chart(spec)
        except Exception as e:
            st.warning(f"Chart skipped – {e}")

        # ── Chat interface ──────────────────────────────────────
    st.divider()
    st.markdown("### 💬 Ask questions about your data")

    # replay prior conversation (skip the initial system prompt)
    for msg in st.session_state.get("messages", [])[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # user prompt box
    if prompt := st.chat_input("Ask anything…"):
        # store + echo the user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # call the model (with function-calling enabled)
        response = call_llm(st.session_state.messages, functions=FUNCTIONS_SPEC)
        msg = response.choices[0].message

        if msg.function_call:               # model chose a helper
            func_name = msg.function_call.name
            args = json.loads(msg.function_call.arguments or "{}")

            if func_name == "aggregate":
                table = aggregate(**args)
            elif func_name == "get_rows":
                table = get_rows(**args)
            else:
                table = pd.DataFrame()

            with st.chat_message("assistant"):
                st.markdown(f"### Result of `{func_name}`")
                st.dataframe(table, use_container_width=True)

            st.session_state.messages.append(
                {"role": "assistant",
                 "content": f"Executed `{func_name}` with args `{args}`."}
            )

        else:                               # free-text answer
            content = msg.content
            with st.chat_message("assistant"):
                st.markdown(content)
            st.session_state.messages.append(
                {"role": "assistant", "content": content}
            )

    # ── PDF export of last assistant message ───────────────
    if st.button("Generate PDF of last answer"):
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=11)

            answer_text = st.session_state.messages[-1]["content"]
            for line in textwrap.wrap(answer_text, 100):
                pdf.cell(0, 8, line, ln=True)

            pdf_bytes = BytesIO()
            pdf.output(pdf_bytes)
            st.download_button(
                "Download PDF",
                data=pdf_bytes.getvalue(),
                file_name="analysis.pdf",
                mime="application/pdf",
            )
        else:
            st.warning("Ask a question first to have something to export!")
