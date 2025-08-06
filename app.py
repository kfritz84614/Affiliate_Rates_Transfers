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
from typing import Any, Dict, List

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
    """Read CSV to DataFrame and coerce numerics & dates."""
    df = pd.read_csv(file)

    # Numeric coercion – only columns that *look* numeric
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        sample = df[col].head(20).astype(str).str.replace(r"[\s,$%()]", "", regex=True)
        if (sample.str.match(r"^-?\d*[\.,]?\d+$").mean()) > 0.8:
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

# ── LLM helper functions exposed to GPT ──────────────────────────────────

def aggregate(by: str, target: str | None = None, metric: str = "sum", top_n: int | None = None) -> pd.DataFrame:
    """Grouped stats: sum | mean | count."""
    df = st.session_state.df
    if target is None:
        target = df.columns[0]
    agg_map = {"sum": "sum", "mean": "mean", "count": "count"}
    if metric not in agg_map:
        metric = "sum"
    if metric == "count":
        ser = df.groupby(by)[target].count()
    else:
        ser = df.groupby(by)[target].agg(metric)
    out = ser.reset_index().rename(columns={target: metric})
    if top_n:
        out = out.nlargest(top_n, metric)
    return out

def get_rows(where: str, columns: List[str] | None = None, limit: int = 20) -> pd.DataFrame:
    """Return rows matching a pandas‑query expression."""
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

# ── LLM wrapper ─────────────────────────────────────────────────────────

def call_llm(messages: List[Dict[str, Any]], functions: List[Dict[str, Any]] | None = None):
    for attempt in range(5):
        try:
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                functions=functions,
            )
            _add_cost(resp.usage.total_tokens)  # type: ignore[attr-defined]
            return resp
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
    raise RuntimeError("OpenAI API failed after retries")

# ── Sidebar: upload & cost ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📤 Upload CSV")
    up = st.file_uploader("Choose file", type=["csv"])
    if up:
        st.session_state.df = load_csv(up)
        st.success("File loaded & cleaned!")
        st.session_state.messages = [
            {"role": "system", "content": "You are a data analyst. Use helper functions when appropriate."}
        ]
        st.session_state.pop("_charts_rendered", None)
        st.session_state.pop("_chart_specs", None)
    st.markdown(f"**Usage cost:** ${st.session_state._cost_usd:.4f}")

# ── Helpers for UI ──────────────────────────────────────────────────────

def show_kpis(df: pd.DataFrame):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    num_cols = df.select_dtypes("number").columns
    if not num_cols.empty:
        c2.metric("Σ of first numeric col", f"{df[num_cols[0]].sum():,.0f}")
        overruns = num_cols.intersection([c for c in df.columns if "overrun" in c.lower()])
        if not overruns.empty:
            c3.metric("Over‑run share", f"{(df[overruns[0]]>0).mean()*100:.1f}%")

def render_chart(spec: Dict[str, str]):
    ct = spec.get("type", "bar")
    x, y, agg = spec.get("x"), spec.get("y"), spec.get("agg", "sum")
    df = st.session_state.df
    if not (x and y):
        st.warning("Chart spec missing x or y – skipped")
        return
    if agg == "count":
        df_plot = df.groupby(x)[y].count().reset_index(name="count"); y = "count"
    else:
        df_plot = df.groupby(x)[y].agg(agg).reset_index()
    fig = {
        "line": px.line,
        "scatter": px.scatter,
        "heatmap": lambda _df, **kw: px.density_heatmap(df, x=x, y=y),
    }.get(ct, px.bar)(df_plot, x=x, y=y)
    st.plotly_chart(fig, use_container_width=True)

# ── Main page ───────────────────────────────────────────────────────────

if df := st.session_state.get("df"):
    # Data loaded – full UI follows

    show_kpis(df)

    # First‑load chart suggestions
    if "_charts_rendered" not in st.session_state:
        prompt = "Provide up to 5 useful chart specs (JSON list) for this dataset."
        resp = call_llm([{"role": "user", "content": prompt}])
        try:
            st.session_state._chart_specs = json.loads(resp.choices[0].message.content)[:5]
        except Exception:
            st.session_state._chart_specs = []
        st.session_state._charts_rendered = True

    for s in st.session_state.get("_chart_specs", []):
        try:
            render_chart(s)
        except Exception as e:
            st.warning(f"Chart skipped – {e}")

    # ── Chat interface ─────────────────────────────────────
    st.divider(); st.markdown("### 💬 Ask questions about your data")

    for m in st.session_state.get("messages", [])[1:]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if q := st.chat_input("Ask anything…"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"): st.markdown(q)

        resp = call_llm(st.session_state.messages, functions=FUNCTIONS_SPEC)
        m = resp.choices[0].message

        if m.function_call:
            fn = m.function_call.name; args = json.loads(m.function_call.arguments or "{}")
            tbl = aggregate(**args) if fn == "aggregate" else get_rows(**args)
            with st.chat_message("assistant"):
                st.markdown(f"### Result of `{fn}`")
                st.dataframe(tbl, use_container_width=True)
            st.session_state.messages.append({"role": "assistant", "content": f"Executed `{fn}` with args `{args}`."})
        else:
            ans = m.content
            with st.chat_message("assistant"): st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

    # ── PDF export button ─────────────────────────────────
    st.divider()
    if st.button("Generate PDF of last answer"):
        msgs = st.session_state.get("messages", [])
        if msgs and msgs[-1]["role"] == "assistant":
            pdf = FPDF(); pdf.set_auto_page_break(True, 15); pdf.add_page(); pdf.set_font("Arial", size=11)
            for line in textwrap.wrap(msgs[-1]["content"], 100):
                pdf.cell(0, 8, line, ln=True)
            buf = BytesIO(); pdf.output(buf)
            st.download_button("Download PDF", buf.getvalue(), "analysis.pdf", "application/pdf")
        else:
            st.warning("Run a query first so there’s something to export!")
else:
    # No data yet – friendly landing message
    st.markdown("### 📂 No data loaded")
    st.markdown("Upload a CSV using the sidebar on the left to get started.")
    st.markdown("The app will clean your data automatically, suggest charts, and let you ask questions in natural language.")
