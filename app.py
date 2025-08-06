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

# ────────────────────────  Auto‑insight visualisations  ──────────────── #

def show_auto_insights(df: pd.DataFrame):
    """Render quick charts that spark strategic thinking."""
    with st.expander("🔍 Auto Insights & Charts", expanded=True):
        num_cols = df.select_dtypes("number").columns.tolist()
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                title="Correlation heat‑map (numeric columns)",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Pick first numeric for category bars
        if num_cols:
            target = num_cols[0]
            cat_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique() <= 20]
            for c in cat_cols[:3]:  # limit to three charts
                agg = (
                    df.groupby(c)[target]
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
                    .rename({target: "Total"}, axis=1)
                )
                fig = px.bar(
                    agg,
                    x=c,
                    y="Total",
                    title=f"Total {target} by {c}",
                    text_auto=".2s",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────  Streamlit UI  ────────────────────────── #

st.set_page_config(page_title="CSV Insight", layout="wide")

st.title("📊 CSV Insight v8·4")

uploaded = st.file_uploader("Drop a CSV", type=["csv"])
if uploaded:
    file_sha = hashlib.md5(uploaded.getvalue()).hexdigest()
    if st.session_state.get("file_sha") != file_sha:
        st.session_state["df"] = load_dataframe(uploaded)
        st.session_state["file_sha"] = file_sha

    df = st.session_state["df"]

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

    # ── Automated insight charts ─────────────────────────────────────
    show_auto_insights(df)

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
