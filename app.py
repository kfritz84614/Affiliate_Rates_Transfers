import streamlit as st
import openai
import pandas as pd
import io

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# UI: Upload CSV
st.title("Affiliate Overrun Analyzer")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Ask question
    question = st.text_area("Ask a question about your data:")

    if st.button("Ask") and question:
        # Trim data sample for prompt (e.g., first 10 rows only)
        sample_csv = df.head(10).to_csv(index=False)

        prompt = f"""You're a data analyst. Here's a table of data:\n\n{sample_csv}\n\nAnswer this question: {question}"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        st.write("### Answer:")
        st.write(response.choices[0].message.content)
