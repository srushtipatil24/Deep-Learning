import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain(vector_store):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant specialized in answering questions from PDF documents.

Instructions:
- Answer ONLY from the provided context.
- If the answer exists, provide a clear, accurate, and detailed response.
- If the exact answer is not available, provide the closest relevant information from the context.
- Only if absolutely nothing relevant exists, reply exactly:
  "I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
""")

    parser = StrOutputParser()
    chain = prompt | llm | parser

    def ask_question(question: str):
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            return {
                "answer": "No relevant information found in the uploaded documents.",
                "source_documents": []
            }

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": answer.strip(),
            "source_documents": retrieved_docs
        }

    return ask_question