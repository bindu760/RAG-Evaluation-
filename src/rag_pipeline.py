
from pathlib import Path

from src.document_loader import load_pdf
from src.document_chunker import split_documents
from src.embedding_model import get_embedding_model
from src.vector_store import create_vector_store
from src.llm_model import get_llm
from src.rag_chain import get_rag_answer


def build_rag_pipeline(pdf_path):
    """
    Load a PDF, split it into chunks,
    create embeddings and build a FAISS vector store.
    """

    # 1. Load PDF
    documents = load_pdf(pdf_path)

    if not documents:
        raise ValueError("No content found in the PDF.")

    # 2. Split documents into chunks
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks generated from the PDF.")

    # 3. Create embedding model
    embedding_model = get_embedding_model()

    # 4. Create FAISS vector store
    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    # 5. Initialize LLM
    llm = get_llm()

    return vector_store, llm, len(chunks)


def ask_pdf_question(question, vector_store, llm, k=3):
    """
    Ask a question about the uploaded PDF.
    """

    result = get_rag_answer(
        question,
        vector_store,
        llm,
        k=k
    )

    return result
