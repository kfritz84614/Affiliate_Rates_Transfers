import streamlit as st
import openai
import pandas as pd
import io
import matplotlib.pyplot as plt

# Set your OpenAI API key securely from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Page config
st.set_page_config(page_title="Affiliate Rate Auditor", layout="wide")
st.title("📊 Affiliate Rate Audit — Ask ChatGPT Anything")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV loaded successfully.")
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head(20), use_container_width=True)

    # Optional: basic column summary
    with st.expander("📌 Column Info"):
        st.write(df.dtypes)

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
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message.content

        st.write("### 💡 Answer from ChatGPT:")
        st.markdown(answer)

    # Optional: automatic chart if the question implies it
    if any(kw in question.lower() for kw in ["chart", "graph", "plot"]):
        st.subheader("📈 Chart Based on Your Data")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-Axis", options=numeric_cols, index=0)
            y_col = st.selectbox("Y-Axis", options=numeric_cols, index=1)
            fig, ax = plt.subplots()
            df.plot(kind="scatter", x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns to generate a plot.")
