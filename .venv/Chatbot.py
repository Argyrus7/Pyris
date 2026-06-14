import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Pyris",
    page_icon="📄"
)

st.header("Pyris")
st.caption("by Argyrus")
st.write("Upload your PDF and ask questions about its content.")

# ==========================================
# GOOGLE API KEY
# ==========================================

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    st.error("Google API key not found.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )

# ==========================================
# PDF PROCESSING
# ==========================================

if file is not None:

    try:
        pdf_reader = PdfReader(file)

        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.error("No readable text found in this PDF.")
            st.stop()

        st.success(
            f"PDF loaded successfully ({len(text):,} characters)"
        )

        user_question = st.text_input(
            "Ask a question about your PDF"
        )

        if user_question:

            with st.spinner("Analyzing PDF..."):

                # Prevent context overflow
                max_chars = 500000
                pdf_content = text[:max_chars]

                prompt = f"""
You are a PDF assistant.

Answer the user's question using ONLY the information
contained in the PDF content below.

If the answer is not present in the PDF, say:
"I could not find that information in the document."

PDF CONTENT:
{pdf_content}

QUESTION:
{user_question}

ANSWER:
"""

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0
                )

                response = llm.invoke(prompt)

                st.subheader("Answer")
                st.write(response.content)

    except Exception as e:
        st.error(f"Error: {str(e)}")
