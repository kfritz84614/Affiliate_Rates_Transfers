import streamlit as st
import pandas as pd
import openai, time, random, logging, json, hashlib
import plotly.express as px
from fpdf import FPDF

OPENAI_MODEL = "gpt-4o-mini"

# ──────────────────────────  OpenAI helper  ────────────────────────── #

def chat_with_retry(**kwargs):
    """Call OpenAI with exponential back‑off on HTTP 429."""
    for attempt in range(6):
        try:
            return openai.chat.completions.create(**kwargs)
        except Exception as e:
            if getattr(e, "status", None) != 429:
                raise
            wait = 2 ** attempt + random.random()
            logging.warning("Rate‑limited – retrying in %.1fs…", wait)
            time.sleep(wait)
    raise RuntimeError("OpenAI retries exhausted")

# ───────────────────────────  Data utilities  ───────────────────────── #

@st.cache_data(show_spinner=False)
def to_number(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"[^0-9.\-()]", "", regex=True)
        .str.replace(r"\((.*)\)", r"-\1", regex=True)
        .astype(float)
        .round(2)
    )

@st.cache_data(show_spinner=False)
def load_dataframe(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    for col in df.select_dtypes(include="object"):
        try:
            df[col] = to_number(df[col])
        except Exception:
            pass
    return df

@st.cache_data(show_spinner=False)
def aggregate(df: pd.DataFrame, by: str, target: str, metric: str = "mean", top_n: int | None = None):
    res = (
        df.groupby(by)[target]
        .agg({"mean": "mean", "median": "median", "sum": "sum"}.get(metric, "mean"))
        .sort_values(ascending=False)
    )
    if top_n:
        res = res.head(int(top_n))
    return res.reset_index().rename({target: "Result"}, axis=1)

@st.cache_data(show_spinner=False)
def get_rows(df: pd.DataFrame, where: str, columns: list[str] | None = None, limit: int = 5):
    view = df.query(where)
    if columns:
        view = view[columns]
    return view.head(limit)

# ─────────────────────  LLM‑guided chart suggestions  ─────────────────── #

@st.cache_data(show_spinner=False)
def suggest_charts(df: pd.DataFrame, n: int = 5):
    """Ask GPT‑4o for up to `n` chart specs helpful for business insight."""
    summary = {
        "n_rows": len(df),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "numeric_summary": df.describe(include="number").round(2).to_dict(),
        "categorical_nunique": {c: int(df[c].nunique()) for c in df.select_dtypes("object").columns},
    }

    fn_schema = {
        "name": "chart_suggestions",
        "description": "Return up to 5 helpful chart specs",
        "parameters": {
            "type": "object",
            "properties": {
                "charts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["bar", "line", "scatter", "heatmap"]},
                            "x": {"type": "string"},
                            "y": {"type": "string"},
                            "metric": {"type": "string", "enum": ["sum", "mean", "count", "none"], "default": "none"},
                            "title": {"type": "string"},
                        },
                        "required": ["type", "x", "y"],
                    },
                    "minItems": 1,
                    "maxItems": 5,
                }
            },
            "required": ["charts"],
        },
    }

    resp = chat_with_retry(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a senior data analyst. Based on the dataframe summary, suggest up to five charts that would help leaders make strategic decisions."},
            {"role": "user", "content": json.dumps(summary)},
        ],
        functions=[fn_schema],
        function_call={"name": "chart_suggestions"},
        temperature=0.2,
    )

    try:
        charts = json.loads(resp.choices[0].message.function_call.arguments)["charts"]
    except Exception:
        charts = []
    return charts[:n]


def render_chart(df: pd.DataFrame, spec: dict):
    ct, x, y = spec["type"], spec["x"], spec["y"]
    metric = spec.get("metric", "none")
    title = spec.get("title") or f"{ct.title()} of {y} by {x}"

    if ct == "bar":
        data = (
            df.groupby(x)[y].agg(metric).reset_index().rename({y: "value"}, axis=1)
            if metric != "none" else df
        )
        fig = px.bar(data, x=x, y="value" if metric != "none" else y, title=title, text_auto=".2s")
    elif ct == "line":
        data = (
            df.groupby(x)[y].agg(metric).reset_index().rename({y: "value"}, axis=1)
            if metric != "none" else df
        )
        fig = px.line(data, x=x, y="value" if metric != "none" else y, title=title)
    elif ct == "scatter":
        fig = px.scatter(df, x=x, y=y, title=title)
    elif ct == "heatmap":
        fig = px.density_heatmap(df, x=x, y=y, title=title)
    else:
        return
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────  Streamlit UI  ───────────────────────────── #

st.set_page_config(page_title="CSV Insight", layout="wide")

st.title("📊 CSV Insight v8·5 – LLM‑Recommended Charts")

uploaded = st.file_uploader("Drop a CSV", type=["csv"])
if uploaded:
    file_sha = hashlib.md5(uploaded.getvalue()).hexdigest()
    if st.session_state.get("file_sha") != file_sha:
        st.session_state.clear()  # reset prior state
        st.session_state["df"] = load_dataframe(uploaded)
        st.session_state["file_sha"] = file_sha
        st.session_state["chart_specs"] = suggest_charts(st.session_state["df"], n=5)

    df = st.session_state["df"]
    chart_specs = st.session_state.get("chart_specs", [])

    # ── Headline KPIs ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    num_cols = df.select_dtypes("number").columns
    if num_cols.any():
        c2.metric("Total", f"{df[num_cols].sum().sum():,.0f}")
        c3.metric("Mean", f"{df[num_cols].mean().mean():,.2f}")
        overruns = df.filter(regex="over.?run", axis=1)
        share = (overruns.gt(0).mean().mean()*100) if not overruns.empty else 0
        c4.metric("Over‑run share", f"{share:.1f}%")

       # ── LLM-recommended charts ──────────────────────────────────────
    if chart_specs:
        with st.expander("📈 Recommended Charts", expanded=True):
            for spec in chart_specs:
                render_chart(df, spec)

    # ── Chat input section (aggregate / rows) ────────────────────────
    prompt = st.chat_input("Ask anything about your data…")
    if prompt:
        sys = (
            "You are a data analyst. Functions: aggregate, get_rows. "
            "Dataframe description:\n" + df.describe(include='all').to_string()
        )
        resp = chat_with_retry(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ],
            functions=[
                {
                    "name": "aggregate",
                    "description": "Group & summarise a numeric column",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "by": {"type": "string"},
                            "target": {"type": "string"},
                            "metric": {"type": "string", "enum": ["mean", "median", "sum"]},
                            "top_n": {"type": "integer", "minimum": 1},
                        },
                        "required": ["by", "target"],
                    },
                },
                {
                    "name": "get_rows",
                    "description": "Return raw rows matching a condition",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "where": {"type": "string"},
                            "columns": {"type": "array", "items": {"type": "string"}},
                            "limit": {"type": "integer", "minimum": 1},
                        },
                        "required": ["where"],
                    },
                },
            ],
            function_call="auto",
            temperature=0.2,
        )

        m = resp.choices[0].message
        if m.function_call:
            fn = m.function_call.name
            args = json.loads(m.function_call.arguments or "{}")
            if fn == "aggregate":
                out = aggregate(df, **args)
                st.session_state["last_answer"] = out.to_string()
                st.dataframe(out)
                if not out.empty:
                    fig = px.bar(
                        out,
                        x=out.columns[0],
                        y="Result",
                        text_auto=".2s",
                        title=f"{args.get('metric','mean').title()} of {args['target']} by {args['by']}",
                        height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            elif fn == "get_rows":
                out = get_rows(df, **args)
                st.session_state["last_answer"] = out.to_string()
                st.dataframe(out)
        else:
            st.session_state["last_answer"] = m.content
            st.write(m.content)

    # ── PDF export of last answer ────────────────────────────────────
    if st.button("Generate PDF of last answer") and st.session_state.get("last_answer"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        for line in st.session_state["last_answer"].splitlines():
            pdf.multi_cell(0, 5, txt=line)
        fname = "answer.pdf"
        pdf.output(fname)
        with open(fname, "rb") as f:
            st.download_button("Download PDF", f, file_name=fname, mime="application/pdf")
