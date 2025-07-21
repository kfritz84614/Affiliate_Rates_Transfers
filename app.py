import streamlit as st
import openai
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt

# Setup
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Connect to Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["GSPREAD_SERVICE_ACCOUNT"], scope)
client = gspread.authorize(credentials)

sheet_url = st.text_input("Google Sheet URL")
question = st.text_area("Ask a question about your data:")

if sheet_url:
    sheet = client.open_by_url(sheet_url).sheet1
    data = pd.DataFrame(sheet.get_all_records())

    st.dataframe(data)

    if st.button("Ask"):
        prompt = f"""You're a data analyst. Given this table:\n\n{data.head(10).to_csv(index=False)}\n\nAnswer this question: {question}"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = response.choices[0].message.content
        st.write("### Answer:")
        st.write(answer)

        # Optional: Create charts based on question keywords
        if "chart" in question.lower() or "graph" in question.lower():
            st.line_chart(data.select_dtypes(include='number'))
