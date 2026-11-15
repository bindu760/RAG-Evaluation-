
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    """
    Create a FAISS vector database
    from document chunks.
    """

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store


def search_documents(vector_store, query, k=3):
    """
    Search for the most relevant documents.
    """

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results
