import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_DB_DIR = "faiss_db"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(chunks):
    os.makedirs(FAISS_DB_DIR, exist_ok=True)
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    vector_store.save_local(FAISS_DB_DIR)
    return vector_store


def load_vector_store():
    embeddings = get_embedding_model()
    return FAISS.load_local(
        FAISS_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )