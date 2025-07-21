import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# Set OpenAI API key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Page setup
st.set_page_config(page_title="Affiliate Rate Auditor", layout="wide")
st.title("Affiliate Rate Audit - Ask ChatGPT Anything")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully.")
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head(20), use_container_width=True)

    # Optional: basic column summary
    with st.expander("Column Info"):
        st.write(df.dtypes)

    # Flexible column name detection
    affiliate_col = next((col for col in df.columns if "affiliate" in col.lower()), None)
    estimated_col = next((col for col in df.columns if "estimate" in col.lower()), None)
    final_col = next((col for col in df.columns if "final" in col.lower() or "actual" in col.lower()), None)

    # Always display Affiliate Cost Comparison Chart if relevant columns exist
    if affiliate_col and estimated_col and final_col:
        st.subheader("Affiliate Cost Comparison")

        summary_df = df.groupby(affiliate_col)[[estimated_col, final_col]].sum().reset_index()
        summary_df = summary_df.melt(id_vars=affiliate_col, value_vars=[estimated_col, final_col], 
                                     var_name="Cost Type", value_name="Total Cost")

        fig = px.bar(summary_df, x=affiliate_col, y="Total Cost", color="Cost Type", barmode="overlay",
                     title="Total Final vs Estimated Cost by Affiliate")

        st.plotly_chart(fig, use_container_width=True)

    # Ask natural language question
    question = st.text_area("What would you like to ask about this data?", placeholder="E.g. Which affiliates overcharge the most?")

    if st.button("Ask ChatGPT"):
        # Use dataset summaries instead of a few sample rows
        summary_stats = df.describe(include='number').astype(str).to_string()
        column_list = [str(col) for col in df.columns.tolist()]

        if affiliate_col and estimated_col and final_col:
            aff_summary = str(df.groupby(affiliate_col)[[estimated_col, final_col]].agg(["count", "sum", "mean"]))
        else:
            aff_summary = "Affiliate/cost column(s) missing."

        prompt = f"""
You are a data analyst auditing affiliate pricing. The full dataset includes the following columns:
{column_list}

Summary Statistics of Numeric Fields:
{summary_stats}

Affiliate Cost Breakdown:
{aff_summary}

Based on the above, answer this question:
{question}
"""

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message.content

        st.write("### Answer from ChatGPT:")
        st.markdown(answer)

    # Optional: user-generated custom chart if implied by question
    if any(kw in question.lower() for kw in ["chart", "graph", "plot"]):
        st.subheader("Chart Based on Your Data")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis", numeric_cols, index=0)
            with col2:
                y_axis = st.selectbox("Select Y-axis", numeric_cols, index=1)

            chart_type = st.radio("Chart Type", ["Scatter", "Bar", "Line"])

            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
            elif chart_type == "Line":
                fig = px.line(df.sort_values(by=x_axis), x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns to generate a chart.")
