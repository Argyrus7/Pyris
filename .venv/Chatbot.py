import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS

# ==========================================
# CONFIG
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

# For Streamlit Cloud, put your key in Secrets:
# GOOGLE_API_KEY="your_key"

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
            st.error(
                "No readable text found in this PDF."
            )
            st.stop()

        # ==========================================
        # TEXT SPLITTING
        # ==========================================

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n", " ", ""]
        )

        chunks = text_splitter.split_text(text)

        if not chunks:
            st.error("No text chunks generated.")
            st.stop()

        # ==========================================
        # EMBEDDINGS
        # ==========================================

        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001"
            )

            # Test embedding immediately
            embeddings.embed_query("test")

        except Exception as e:
            st.error(f"Embedding Model Error: {str(e)}")
            st.stop()

        # ==========================================
        # VECTOR STORE
        # ==========================================

        try:
            vector_store = FAISS.from_texts(
                chunks,
                embeddings
            )

        except Exception as e:
            st.error(f"FAISS Error: {str(e)}")
            st.stop()

        # ==========================================
        # QUESTION INPUT
        # ==========================================

        user_question = st.text_input(
            "Ask a question about your PDF"
        )

        if user_question:

            docs = vector_store.similarity_search(
                user_question,
                k=4
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
You are a helpful PDF assistant.

Answer ONLY using information found in the context below.

Context:
{context}

Question:
{user_question}

Answer:
"""

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0
            )

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)

    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")

