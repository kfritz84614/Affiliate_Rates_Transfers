"""
Streamlit CSV Explorer – v9.1 (2025‑08‑06)
------------------------------------------------
• Case‑insensitive column matching ("vehicle" == "Vehicle")
• Graceful fallback when LLM suggests unknown columns
• Fixed duplicate `st.plotly_chart` call
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

st.set_page_config(page_title="CSV Explorer AI", layout="wide", initial_sidebar_state="expanded")

openai.api_key = st.secrets.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"
TOKEN_PRICE_USD = 0.00001  # ≈ $0.01 per 1k tokens – adjust when pricing changes
st.session_state.setdefault("_cost_usd", 0.0)


def _add_cost(tokens: int) -> None:
    st.session_state._cost_usd += tokens * TOKEN_PRICE_USD / 1000

# ── Utils ───────────────────────────────────────────────────────────────

def resolve_col(name: str | None, df: pd.DataFrame) -> str | None:
    """Return real column name ignoring case/extra spaces."""
    if not name:
        return None
    name_s = name.strip().lower()
    mapping = {c.lower(): c for c in df.columns}
    return mapping.get(name_s)

# ── Data loading & coercion ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
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
    for col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df

# ── LLM helper funcs exposed via function calling ───────────────────────

def aggregate(by: str, target: str | None = None, metric: str = "sum", top_n: int | None = None) -> pd.DataFrame:
    df = st.session_state.df
    by_real = resolve_col(by, df)
    if by_real is None:
        st.warning(f"Column '{by}' not found – available: {list(df.columns)[:10]}…")
        return pd.DataFrame()
    target_real = resolve_col(target, df) if target else None
    if target and target_real is None:
        st.warning(f"Target column '{target}' not found – using first numeric col.")
        num_cols = df.select_dtypes("number").columns
        target_real = num_cols[0] if not num_cols.empty else by_real
    if target_real is None:
        target_real = by_real
    if metric == "count":
        ser = df.groupby(by_real)[target_real].count()
    elif metric == "mean":
        ser = df.groupby(by_real)[target_real].mean()
    else:
        ser = df.groupby(by_real)[target_real].sum()
    out = ser.reset_index().rename(columns={target_real: metric})
    if top_n:
        out = out.nlargest(top_n, metric)
    return out

def get_rows(where: str, columns: List[str] | None = None, limit: int = 20) -> pd.DataFrame:
    df = st.session_state.df
    try:
        subset = df.query(where)
    except Exception as e:
        st.warning(f"❌ Bad query: {e}")
        return pd.DataFrame()
    if columns:
        real_cols = [c for col in columns if (c := resolve_col(col, df))]
        subset = subset[real_cols]
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

# ── OpenAI wrapper with retries & cost logging ──────────────────────────

def call_llm(msgs: List[Dict[str, Any]], *, funcs: List[Dict[str, Any]] | None = None):
    for attempt in range(5):
        try:
            r = openai.chat.completions.create(model=MODEL, messages=msgs, functions=funcs)
            _add_cost(r.usage.total_tokens)  # type: ignore[attr-defined]
            return r
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
    raise RuntimeError("OpenAI failed repeatedly")

# ── Sidebar: upload CSV + cost ──────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📤 Upload CSV")
    up = st.file_uploader("Choose file", type=["csv"])
    if up:
        st.session_state.df = load_csv(up)
        st.success("File loaded & cleaned!")
        st.session_state.messages = [{"role": "system", "content": "You are a data analyst. Use helper functions when appropriate."}]
        st.session_state.pop("_charts_rendered", None)
        st.session_state.pop("_chart_specs", None)
    st.markdown(f"**Usage cost:** ${st.session_state._cost_usd:.4f}")

# ── UI helpers ──────────────────────────────────────────────────────────

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
    df = st.session_state.df
    ct = spec.get("type", "bar")
    x = resolve_col(spec.get("x"), df)
    y = resolve_col(spec.get("y"), df)
    agg = spec.get("agg", "sum")
    if not x or not y:
        st.warning("Chart spec has unknown columns – skipped")
        return
    try:
        if agg == "count":
            df_plot = df.groupby(x)[y].count().reset_index(name="count"); y = "count"
        else:
            df_plot = df.groupby(x)[y].agg(agg).reset_index()
    except Exception as e:
        st.warning(f"Chart aggregation failed: {e}")
        return
    fig_fn = {
        "line": px.line,
        "scatter": px.scatter,
        "heatmap": lambda _df, **kw: px.density_heatmap(df, x=x, y=y),
    }.get(ct, px.bar)
    fig = fig_fn(df_plot, x=x, y=y)
    st.plotly_chart(fig, use_container_width=True)

# ── Main area ───────────────────────────────────────────────────────────

df = st.session_state.get("df")
if df is not None:
    show_kpis(df)

    # First‑load chart suggestions
    if "_charts_rendered" not in st.session_state:
        prompt = "Provide up to 5 useful chart specs (JSON list) for this dataset. Keys: type, x, y, agg."
        try:
            r = call_llm([{"role": "user", "content": prompt}])
            charts = json.loads(r.choices[0].message.content)
        except Exception:
            charts = []
        if not charts:
            cat = df.select_dtypes("object").columns
            num = df.select_dtypes("number").columns
            if cat.any() and num.any():
                charts = [{"type": "bar", "x": cat[0], "y": num[0], "agg": "sum"}]
        st.session_state._chart_specs = charts[:5]
        st.session_state._charts_rendered = True

    for c in st.session_state.get("_chart_specs", []):
        render_chart(c)

        # ── Chat interface ─────────────────────────────────────
    st.divider()
    st.markdown("### 💬 Ask questions about your data")

    # replay prior conversation (skip the initial system prompt)
    for m in st.session_state.get("messages", [])[1:]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # user prompt box
    if q := st.chat_input("Ask anything…"):
        # store + echo the user message
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        # call the model (function-calling enabled)
        resp = call_llm(st.session_state.messages, funcs=FUNCTIONS_SPEC)
        msg = resp.choices[0].message

        if msg.function_call:                       # model chose a helper
            fn   = msg.function_call.name
            args = json.loads(msg.function_call.arguments or "{}")

            if fn == "aggregate":
                result_df = aggregate(**args)
            elif fn == "get_rows":
                result_df = get_rows(**args)
            else:
                result_df = pd.DataFrame()

            with st.chat_message("assistant"):
                st.markdown(f"### Result of `{fn}`")
                st.dataframe(result_df, use_container_width=True)

            st.session_state.messages.append(
                {"role": "assistant",
                 "content": f"Executed `{fn}` with args {args}."}
            )

        else:                                       # free-text answer
            answer = msg.content
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

    # ── PDF export of last assistant message ───────────────
    st.divider()
    if st.button("Generate PDF of last answer"):
        msgs = st.session_state.get("messages", [])
        if msgs and msgs[-1]["role"] == "assistant":
            pdf = FPDF()
            pdf.set_auto_page_break(True, 15)
            pdf.add_page()
            pdf.set_font("Arial", size=11)

            for line in textwrap.wrap(msgs[-1]["content"], 100):
                pdf.cell(0, 8, line, ln=True)

            buf = BytesIO()
            pdf.output(buf)

            st.download_button(
                label="Download PDF",
                data=buf.getvalue(),
                file_name="analysis.pdf",
                mime="application/pdf",
            )
        else:
            st.warning("Ask a question first so there’s something to export!")

else:
    # No CSV yet – friendly landing screen
    st.markdown("### 📂 No data loaded")
    st.markdown(
        "Upload a CSV using the sidebar to start exploring your data with "
        "automatic charts and AI-powered Q&A."
    )
