import streamlit as st
import pandas as pd
import openai, time, random, logging, json, hashlib
import plotly.express as px
from fpdf import FPDF

OPENAI_MODEL = "gpt-4o-mini"

# ──────────────────────────  OpenAI helper  ────────────────────────── #

def chat_with_retry(**kwargs):
    """Call OpenAI with exponential back‑off on HTTP 429."""
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
    """Convert strings like "$1,234" or "(123)" to floats."""
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
            pass  # non‑numeric text
    return df

@st.cache_data(show_spinner=False)
def aggregate(df: pd.DataFrame, by: str, target: str, metric: str = "mean", top_n: int | None = None):
    agg_map = {"mean": "mean", "median": "median", "sum": "sum"}
    res = (
        df.groupby(by)[target]
        .agg(agg_map.get(metric, "mean"))
        .sort_values(ascending=False)
    )
    if top_n:
        res = res.head(int(top_n))
    return res.reset_index(names=[by]).rename({target: "Result"}, axis=1)

@st.cache_data(show_spinner=False)
def get_rows(df: pd.DataFrame, where: str, columns: list[str] | None = None, limit: int = 5):
    view = df.query(where)
    if columns:
        view = view[columns]
    return view.head(limit)

# ────────────────────────  LLM‑driven chart suggestions  ─────────────── #

@st.cache_data(show_spinner=False)
def suggest_charts(df: pd.DataFrame):
    """Ask the LLM to suggest up to 3 helpful chart specs."""
    summary = {
        "columns": df.dtypes.astype(str).to_dict(),
        "n_rows": len(df),
        "numeric_summary": df.describe(include="number").round(2).to_dict(),
    }

    functions = [
        {
            "name": "chart_suggestions",
            "description": "Return up to 3 chart specifications helpful for business decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "charts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["bar", "line", "scatter", "heatmap"],
                                },
                                "x": {"type": "string"},
                                "y": {"type": "string"},
                                "metric": {
                                    "type": "string",
                                    "enum": ["sum", "mean", "count", "none"],
                                    "default": "none",
                                },
                                "title": {"type": "string"},
                            },
                            "required": ["type", "x", "y"],
                        },
                        "minItems": 1,
                        "maxItems": 3,
                    }
                },
                "required": ["charts"],
            },
        }
    ]

    system = (
        "You are an analytics expert. Based on the dataframe summary provided, "
        "suggest up to three charts that would give strategic business insight. "
        "Use simple column names; if aggregation needed, specify metric (sum / mean / count)."
    )

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(summary)},
    ]

    resp = chat_with_retry(
        model=OPENAI_MODEL,
        messages=msgs,
        functions=functions,
        function_call={"name": "chart_suggestions"},
        temperature=0.2,
    )

    charts = json.loads(resp.choices[0].message.function_call.arguments)["charts"]
    return charts


def render_chart(df: pd.DataFrame, spec: dict):
    """Render a Plotly chart based on the spec."""
    chart_type = spec["type"]
    x = spec["x"]
    y = spec["y"]
    metric = spec.get("metric", "none")
    title = spec.get("title", f"{chart_type.title()} of {y} by {x}")

    if chart_type == "bar":
        if metric != "none":
            data = (
                df.groupby(x)[y]
                .agg(metric)
                .reset_index()
                .rename({y: "value"}, axis=1)
            )
            fig = px.bar(data, x=x, y="value", title=title, text_auto=".2s")
        else:
            fig = px.bar(df, x=x, y=y, title=title, text_auto=".2s")
    elif chart_type == "line":
        if metric != "none":
            data = (
                df.groupby(x)[y]
                .agg(metric)
                .reset_index()
                .rename({y: "value"}, axis=1)
            )
            fig = px.line(data, x=x, y="value", title=title)
        else:
            fig = px.line(df, x=x, y=y, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, title=title)
    elif chart_type == "heatmap":
        fig = px.density_heatmap(df, x=x, y=y, title=title)
    else:
        return
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────  Streamlit UI  ────────────────────────── #

st.set_page_config(page_title="CSV Insight", layout="wide")

st.title("📊 CSV Insight v9·0 – LLM‑guided Charts")

uploaded = st.file_uploader("Drop a CSV", type=["csv"])
if uploaded:
    file_sha = hashlib.md5(uploaded.getvalue()).hexdigest()
    if st.session_state.get("file_sha") != file_sha:
        st.session_state["df"] = load_dataframe(uploaded)
        st.session_state["file_sha"] = file_sha
        # get new suggestions whenever a fresh file appears
        st.session_state["chart_specs"] = suggest_charts(st.session_state["df"])

    df = st.session_state["df"]
    chart_specs = st.session_state.get("chart_specs", [])

    # ── Headline KPIs ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    numeric_cols = df.select_dtypes("number").columns
    if len(numeric_cols):
        col2.metric("Total", f"{df[numeric_cols].sum().sum():,.0f}")
        col3.metric("Mean", f"{df[numeric_cols].mean().mean():,.2f}")
        overruns = df.filter(regex="over.?run", axis=1)
        share = (overruns.gt(0).mean().mean() * 100) if not overruns.empty else 0
        col4.metric("Over‑run share", f"{share:.1f}%")

    # ── LLM‑suggested charts ────────────────────────────────────────
    if chart_specs:
        with st.expander("📈 Suggested Insightful Charts", expanded=True):
            for spec in chart_specs:
                render_chart(df, spec)

    # ── Chat input section ──────────────────────────────────────────
    prompt = st.chat_input("Ask anything about your data…")
    if prompt:
        system = (
            "You are a data analyst. Available functions allow grouping or row‑level retrieval. "
            "The dataframe description is:\n" + df.describe(include='all').to_string()
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        functions = [
            {
                "name": "aggregate",
                "description": "Group and summarise a numeric column",
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
                "description": "Return raw data rows matching a condition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "where": {"type": "string"},
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["where"],
                },
            },
        ]

        resp = chat_with_retry(
            model=OPENAI_MODEL,
            messages=msgs,
            functions=functions,
            function_call="auto",
            temperature=0.2,
        )

        msg = resp.choices[0].message
        if msg.function_call:
            func_name = msg.function_call.name
            args = json.loads(msg.function_call.arguments or "{}")
            if func_name == "aggregate":
                result_df = aggregate(df, **args)
                st.session_state["last_answer"] = result_df.to_string()
                st.dataframe(result_df)
                if not result_df.empty:
                    fig = px.bar(
                        result_df,
                        x=result_df.columns[0],
                        y="Result",
                        title=f"{args.get('metric', 'mean').title()} of {args['target']} by {args['by']}",
                        text_auto=".2s",
                        height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            elif func_name == "get_rows":
                result_df = get_rows(df, **args)
                st.session_state["last_answer"] = result_df.to_string()
                st.dataframe(result_df)
        else:
            st.session_state["last_answer"] = msg.content
            st.write(msg.content)

    # ── PDF export ──────────────────────────────────────────────────
    if st.button("Generate PDF of last answer") and st.session_state.get("last_answer"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        for line in st.session_state["last_answer"].splitlines():
            pdf.multi_cell(0, 5, txt=line)
        fn = "answer.pdf"
        pdf.output(fn)
        with open(fn, "rb") as f:
            st.download_button("Download PDF", f, file_name=fn, mime="application/pdf")
