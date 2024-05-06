import streamlit as st
import requests


SERVER_URL = "http://0.0.0.0:8000/upload-pdf/"


# streamlit application
def main():
    st.title("Research Quest: Q and A using LLMs")

    file = st.file_uploader("Upload a PDF file:", type="pdf")
    fname = st.text_input("Enter file name:")
    query = st.text_input("Enter your query:")

    # on cliking the send button 
    if st.button("Send"):
        if file is None and not fname:
            st.error("Please upload a file or provide a file name")
            return

        if file:
            response = requests.post(SERVER_URL, files={"file": file}, data={"fname": "None", "query": query})
        else:
            response = requests.post(SERVER_URL, data={"fname": fname, "query": query})

        if response.status_code == 200:
            result = response.json()
            fname =  result["fname"]
            st.write("Answer:", result["response"])
        else:
            st.error("Error occurred while processing the request")

if __name__ == "__main__":
    main()
