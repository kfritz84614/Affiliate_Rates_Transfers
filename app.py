import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# Set OpenAI API key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Page setup
st.set_page_config(page_title="Affiliate Rate Auditor", layout="wide")
st.title("\ud83d\udcca Affiliate Rate Audit \u2014 Ask ChatGPT Anything")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully.")
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head(20), use_container_width=True)

    # Optional: basic column summary
    with st.expander("\ud83d\udccc Column Info"):
        st.write(df.dtypes)

    # Ask natural language question
    question = st.text_area("What would you like to ask about this data?", placeholder="E.g. Which affiliates overcharge the most?")

    if st.button("Ask ChatGPT"):
        sample_data = df.head(10).to_csv(index=False)

        prompt = f"""
You are a data analyst auditing affiliate pricing. A CSV was uploaded with the following sample rows:

{sample_data}

Answer the following question based on the full dataset (assume the rest follows the same structure):
{question}
"""

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message.content

        st.write("### \ud83d\udca1 Answer from ChatGPT:")
        st.markdown(answer)

    # Custom Affiliate Cost Comparison Chart
    if "Affiliate ID" in df.columns and "Estimated Cost" in df.columns and "Final Cost" in df.columns:
        st.subheader("\ud83d\udcc8 Affiliate Cost Comparison")

        summary_df = df.groupby("Affiliate ID")[["Estimated Cost", "Final Cost"]].sum().reset_index()
        summary_df = summary_df.melt(id_vars="Affiliate ID", value_vars=["Estimated Cost", "Final Cost"], 
                                     var_name="Cost Type", value_name="Total Cost")

        fig = px.bar(summary_df, x="Affiliate ID", y="Total Cost", color="Cost Type", barmode="overlay",
                     title="Total Final vs Estimated Cost by Affiliate")

        st.plotly_chart(fig, use_container_width=True)

    # Optional: automatic chart if the question implies it
    if any(kw in question.lower() for kw in ["chart", "graph", "plot"]):
        st.subheader("\ud83d\udcc8 Chart Based on Your Data")

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
