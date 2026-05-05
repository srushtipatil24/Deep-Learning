import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

UPLOAD_DIR = "uploads"


def save_uploaded_file(uploaded_file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_multiple_pdfs(file_paths):
    all_documents = []
    for file_path in file_paths:
        documents = load_pdf(file_path)
        all_documents.extend(documents)
    return all_documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)