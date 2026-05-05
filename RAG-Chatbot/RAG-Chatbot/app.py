import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.makedirs("uploads", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)

import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import (
    save_uploaded_file,
    load_multiple_pdfs,
    split_documents
)
from src.vector_store import create_vector_store
from src.rag_chain import get_rag_chain

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI CSS
st.markdown("""
<style>
    #MainMenu, footer, header {
        visibility: hidden;
    }

    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #1e293b 100%);
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .sub-title {
        text-align: center;
        font-size: 1.3rem;
        color: #cbd5e1;
        margin-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stButton > button {
        width: 100%;
        border-radius: 15px;
        padding: 12px;
        font-weight: bold;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.5);
    }

    .stFileUploader {
        border: 2px dashed rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 10px;
        background: rgba(255,255,255,0.05);
    }

    .metric-card {
        background: rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🤖 AI PDF Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload PDFs, Ask Questions, Get Intelligent Answers</div>', unsafe_allow_html=True)

# Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = 0

# Sidebar
with st.sidebar:
    st.markdown("## 📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process Documents"):
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                file_paths = []

                for file in uploaded_files:
                    path = save_uploaded_file(file)
                    file_paths.append(path)

                documents = load_multiple_pdfs(file_paths)
                chunks = split_documents(documents)
                vector_store = create_vector_store(chunks)

                st.session_state.vector_store = vector_store
                st.session_state.rag_chain = get_rag_chain(vector_store)
                st.session_state.processed_files = len(uploaded_files)

            st.success("✅ Documents processed successfully!")
        else:
            st.warning("⚠️ Please upload at least one PDF.")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        f"""
        <div class="metric-card">
            <h3>📄 Files Processed</h3>
            <h1>{st.session_state.processed_files}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
query = st.chat_input("Ask anything about your uploaded PDFs...")

if query:
    if st.session_state.rag_chain is None:
        st.warning("⚠️ Please upload and process PDFs first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain(query)
            answer = response["answer"]
            st.markdown(answer)

            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(response["source_documents"], 1):
                    source = os.path.basename(doc.metadata.get("source", "Unknown"))
                    st.markdown(f"### 📄 Source {i}: {source}")
                    st.write(doc.page_content[:600] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})