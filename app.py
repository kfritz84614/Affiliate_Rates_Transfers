"""
Streamlit CSV Explorer – v9.1 (2025‑08‑06)
------------------------------------------------
• Case‑insensitive column matching ("vehicle" == "Vehicle")
• Graceful fallback when LLM suggests unknown columns
• Fixed duplicate `st.plotly_chart` call
"""

from __future__ import annotations

import os
import io
import json
import re
import math
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CSV Analyst", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI client (expects OPENAI_API_KEY in Streamlit secrets or env)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", None)))
    _OPENAI_READY = _client.api_key is not None
except Exception:
    _client = None
    _OPENAI_READY = False

MODEL_FOR_ANALYSIS = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
_CURRENCY_RE = re.compile(r"[,$%\s]")


def _to_number(x: Any) -> Optional[float]:
    """Robust numeric coercion:
    - Treat parentheses as negatives: (123.45) -> -123.45
    - Strip currency/commas/percent signs
    - Empty/invalid -> None
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = _CURRENCY_RE.sub("", s)
    if s == "":
        return None
    try:
        val = float(s)
        if neg:
            val = -val
        return val
    except ValueError:
        return None


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Trim column names
    df.columns = [str(c).strip() for c in df.columns]

    # Try parsing datetimes for any column that "looks" like dates
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50).str.lower()
            if sample.empty:
                continue
            has_date_hint = (
                sample.str.contains(r"\d{4}-\d{1,2}-\d{1,2}").any()
                or sample.str.contains(r"\d{1,2}/\d{1,2}/\d{2,4}").any()
                or sample.str.contains("am").any()
                or sample.str.contains("pm").any()
                or sample.str.contains("t\d{2}:\d{2}").any()
            )
            if has_date_hint:
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                except Exception:
                    pass  # keep as is if it fails

    # Coerce numeric-looking object columns to numbers (currency, parentheses, etc.)
    for col in df.columns:
        if df[col].dtype == object:
            # only attempt if >60% look numeric/currency
            s = df[col].dropna().astype(str)
            if s.empty:
                continue
            sample = s.sample(min(len(s), 200), random_state=0)
            looks_numeric = sample.str.contains(r"^[\s\($-]?[\d,]+(\.[\d]+)?[%\s\)]?$").mean() > 0.6
            if looks_numeric:
                df[col] = df[col].map(_to_number)

    return df


@st.cache_data(show_spinner=False)
def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[c])]
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    head_rows = df.head(5).to_dict(orient="records")

    # Basic stats for numeric columns
    desc = df[numeric_cols].describe().to_dict() if numeric_cols else {}

    # Top categories for a few categorical columns
    topcats: Dict[str, List[Tuple[str, int]]] = {}
    for c in categorical_cols[:6]:
        vc = (
            df[c]
            .astype(str)
            .replace({"nan": None, "None": None})
            .dropna()
            .value_counts()
            .head(10)
        )
        topcats[c] = [(str(k), int(v)) for k, v in vc.items()]

    sample_rows = min(len(df), 100)
    sample_data = df.sample(sample_rows, random_state=42).to_dict(orient="records") if sample_rows > 0 else []

    return {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "head": head_rows,
        "describe": desc,
        "top_categories": topcats,
        "sample_rows": sample_rows,
        "sample_data": sample_data,
    }


def kpi_block(df: pd.DataFrame):
    n_rows, n_cols = df.shape
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Rows", f"{n_rows:,}")
    kpi_cols[1].metric("Columns", f"{n_cols:,}")

    if numeric_cols:
        totals = df[numeric_cols].sum(numeric_only=True)
        means = df[numeric_cols].mean(numeric_only=True)
        # pick the most relevant numeric column heuristically
        def score(col):
            lc = col.lower()
            return (
                ("amount" in lc) * 5 +
                ("cost" in lc) * 5 +
                ("price" in lc) * 4 +
                ("revenue" in lc) * 5 +
                ("total" in lc) * 3 +
                ("overrun" in lc) * 3
            )
        best = max(numeric_cols, key=score)
        kpi_cols[2].metric(f"Σ {best}", f"{totals[best]:,.2f}")
        kpi_cols[3].metric(f"μ {best}", f"{means[best]:,.2f}")

    # Optional domain-specific: overrun share if columns exist
    def find_col(names: List[str]) -> Optional[str]:
        for cand in df.columns:
            lc = cand.lower()
            if any(name in lc for name in names):
                return cand
        return None

    est_col = find_col(["estimated", "estimate"])
    act_col = find_col(["actual", "final"]) if est_col else None
    if est_col and act_col and pd.api.types.is_numeric_dtype(df[est_col]) and pd.api.types.is_numeric_dtype(df[act_col]):
        overruns = (df[act_col] > df[est_col]).mean()
        st.caption(f"Overrun share ( {act_col} > {est_col} ): **{overruns*100:.1f}%**")


# ──────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ──────────────────────────────────────────────────────────────────────────────
CHART_SPEC_INSTRUCTIONS = {
    "role": "system",
    "content": (
        "You are a data viz planner. Given a dataset summary and optional user request, "
        "return ONLY JSON: a list of 1-5 chart specs. Each spec keys: "
        "type(one of ['bar','line','area','scatter','histogram','box']), "
        "x(str), y(str or list[str]), color(optional str), agg(optional one of ['sum','mean','count','median']), "
        "title(str), orientation(optional 'h'|'v'), stack(optional bool), trendline(optional bool), "
        "filters(optional list of {column:str, op:str, value:any}). "
        "Use only columns that exist. Prefer high-signal metrics like cost, revenue, price, overrun, hours."
    )
}


def llm_suggest_charts(summary: Dict[str, Any], user_request: Optional[str] = None, n: int = 5) -> List[Dict[str, Any]]:
    if not _OPENAI_READY:
        return []
    prompt = {
        "role": "user",
        "content": (
            "DATASET SUMMARY\n" + json.dumps(summary) +
            ("\nUSER REQUEST:\n" + user_request if user_request else "") +
            f"\nReturn JSON array of up to {n} chart specs."
        )
    }
    try:
        resp = _client.chat.completions.create(
            model=MODEL_FOR_ANALYSIS,
            messages=[CHART_SPEC_INSTRUCTIONS, prompt],
            temperature=0.2,
            response_format={"type": "json_object"}  # encourages valid JSON
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        specs = data if isinstance(data, list) else data.get("charts") or data.get("specs") or []
        if not isinstance(specs, list):
            return []
        return specs[:n]
    except Exception:
        # Fallback: try naive JSON extraction
        try:
            text = resp.choices[0].message.content  # type: ignore
            m = re.search(r"\[.*\]", text, re.S)
            if m:
                return json.loads(m.group(0))[:n]
        except Exception:
            pass
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Chart renderer
# ──────────────────────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    for f in filters:
        col = f.get("column")
        op = (f.get("op") or "==").strip()
        val = f.get("value")
        if col not in out.columns:
            continue
        series = out[col]
        try:
            if op == "==":
                out = out[series == val]
            elif op == "!=":
                out = out[series != val]
            elif op == ">":
                out = out[series.astype(float) > float(val)]
            elif op == ">=":
                out = out[series.astype(float) >= float(val)]
            elif op == "<":
                out = out[series.astype(float) < float(val)]
            elif op == "<=":
                out = out[series.astype(float) <= float(val)]
            elif op == "contains":
                out = out[series.astype(str).str.contains(str(val), case=False, na=False)]
        except Exception:
            continue
    return out


def render_chart(df: pd.DataFrame, spec: Dict[str, Any]):
    ctype = (spec.get("type") or "bar").lower()
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    agg = (spec.get("agg") or "sum").lower()
    title = spec.get("title") or ""
    orientation = spec.get("orientation")
    stack = bool(spec.get("stack"))
    trendline = bool(spec.get("trendline"))
    filters = spec.get("filters")

    # Basic validation
    if not x or (isinstance(y, list) and len(y) == 0):
        return

    dff = apply_filters(df, filters)

    # Grouping/aggregation if agg provided
    def agg_df(d: pd.DataFrame) -> pd.DataFrame:
        if y is None:
            return d
        ycols = y if isinstance(y, list) else [y]
        cols = [c for c in ycols if c in d.columns and pd.api.types.is_numeric_dtype(d[c])]
        if not cols:
            return d[[x] + ([color] if color else [])].assign(_ones=1)
        if x not in d.columns:
            return d
        group_cols = [x] + ([color] if color else [])
        grouped = d.groupby(group_cols, dropna=False)[cols]
        if agg == "sum":
            out = grouped.sum().reset_index()
        elif agg == "mean":
            out = grouped.mean().reset_index()
        elif agg == "median":
            out = grouped.median().reset_index()
        elif agg == "count":
            out = grouped.count().reset_index()
        else:
            out = grouped.sum().reset_index()
        return out

    dfx = agg_df(dff)

    # Choose plotly constructor
    if ctype == "bar":
        fig = px.bar(dfx, x=x, y=y, color=color, title=title, orientation=orientation)
        if stack and color:
            fig.update_layout(barmode="stack")
    elif ctype == "line":
        fig = px.line(dfx, x=x, y=y, color=color, title=title)
    elif ctype == "area":
        fig = px.area(dfx, x=x, y=y, color=color, title=title)
    elif ctype == "scatter":
        fig = px.scatter(dfx, x=x, y=y, color=color, title=title, trendline=("ols" if trendline else None))
    elif ctype == "histogram":
        fig = px.histogram(dff, x=x, color=color, title=title)
    elif ctype == "box":
        fig = px.box(dff, x=x, y=y if not isinstance(y, list) else y[0], color=color, title=title)
    else:
        fig = px.bar(dfx, x=x, y=y, color=color, title=title)

    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Domain-specific bar: total actual vs estimated by Affiliate (if columns exist)
# ──────────────────────────────────────────────────────────────────────────────

def maybe_affiliate_est_vs_actual(df: pd.DataFrame):
    # heuristics for column names
    def find_col(cands: List[str]) -> Optional[str]:
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in cands):
                return c
        return None

    affiliate = find_col(["affiliate", "vendor", "partner"])  # id or name
    est = find_col(["estimated", "estimate", "quoted"])      # money
    act = find_col(["actual", "final", "billed"])            # money

    if not (affiliate and est and act):
        return False
    if not (pd.api.types.is_numeric_dtype(df[est]) and pd.api.types.is_numeric_dtype(df[act])):
        return False

    dfx = df[[affiliate, est, act]].copy()
    grp = dfx.groupby(affiliate, dropna=False)[[est, act]].sum().reset_index()
    melted = grp.melt(id_vars=affiliate, value_vars=[act, est], var_name="Kind", value_name="Amount")

    fig = px.bar(
        melted,
        x=affiliate,
        y="Amount",
        color="Kind",
        title="Total Actual vs Estimated by Affiliate",
    )
    fig.update_layout(barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.header("CSV Analyst")
    st.sidebar.caption("Upload → KPIs → LLM analysis → Recommended charts → Ask for more")


def upload_section() -> Optional[pd.DataFrame]:
    file = st.file_uploader("Upload a CSV", type=["csv"])
    if not file:
        return None

    # Robust CSV read (utf-8 as default, fallback to pyarrow if available)
    content = file.read()
    bio = io.BytesIO(content)
    try:
        df = pd.read_csv(bio)
    except Exception:
        bio.seek(0)
        try:
            df = pd.read_csv(bio, engine="pyarrow")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return None

    return clean_dataframe(df)


def llm_section(df: pd.DataFrame, summary: Dict[str, Any]):
    st.subheader("LLM Analysis")
    if not _OPENAI_READY:
        st.info("Set OPENAI_API_KEY in Streamlit secrets or environment to enable LLM features.")
        return

    # Short reasoning if dataset small (<= 2500 rows): include entire CSV as JSON lines (capped by 2500)
    allow_full = len(df) <= 2500

    user_q = st.text_input("Ask a question about the data or request a chart (e.g., 'show top affiliates by overrun'):")
    if st.button("Run Analysis", use_container_width=True) and user_q:
        with st.spinner("Thinking..."):
            messages = [
                {"role": "system", "content": "You are a senior data analyst. Be precise and concise."},
                {"role": "user", "content": (
                    "DATASET SUMMARY\n" + json.dumps(summary) +
                    ("\nFULL DATA JSONL\n" + "\n".join(json.dumps(r) for r in df.to_dict(orient="records")) if allow_full else "") +
                    "\nQUESTION:\n" + user_q
                )}
            ]
            try:
                resp = _client.chat.completions.create(
                    model=MODEL_FOR_ANALYSIS,
                    messages=messages,
                    temperature=0.2,
                )
                answer = resp.choices[0].message.content
                if answer:
                    st.markdown(answer)
            except Exception as e:
                st.error(f"LLM error: {e}")

    st.divider()
    st.subheader("Ask for more charts")
    req = st.text_input("Describe the charts you want (natural language).")
    n = st.slider("How many charts?", 1, 5, 3, 1)
    if st.button("Generate Charts", use_container_width=True, key="charts_nl"):
        with st.spinner("Choosing charts..."):
            specs = llm_suggest_charts(summary, user_request=req, n=n)
        if not specs:
            st.warning("No chart specs returned.")
        for i, spec in enumerate(specs, 1):
            st.markdown(f"**Chart {i}:** {spec.get('title') or ''}")
            try:
                render_chart(df, spec)
            except Exception as e:
                st.error(f"Chart {i} failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    st.title("CSV Analyst")

    df = upload_section()
    if df is None:
        st.info("Upload a CSV to begin.")
        return

    st.success("CSV loaded.")

    # KPIs
    kpi_block(df)

    # Summary (cached)
    summary = summarize_dataframe(df)

    with st.expander("Dataset Preview", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    # LLM‑recommended charts on load (up to 5)
    st.subheader("LLM‑Recommended Charts")
    if _OPENAI_READY:
        with st.spinner("Letting the model pick helpful visuals..."):
            specs = llm_suggest_charts(summary, user_request=None, n=5)
        if not specs:
            st.caption("No recommendations from the model.")
        for i, spec in enumerate(specs, 1):
            st.markdown(f"**Chart {i}:** {spec.get('title') or ''}")
            try:
                render_chart(df, spec)
            except Exception as e:
                st.error(f"Chart {i} failed: {e}")
    else:
        st.caption("Set OPENAI_API_KEY to enable auto‑recommended charts.")

    # Domain-specific affiliate Actual vs Estimated (if columns exist)
    st.subheader("Actual vs Estimated by Affiliate (if available)")
    shown = maybe_affiliate_est_vs_actual(df)
    if not shown:
        st.caption("Suitable columns not found. This section auto-hides when unavailable.")

    # Chat / NL analysis + NL charts
    llm_section(df, summary)


if __name__ == "__main__":
    main()
