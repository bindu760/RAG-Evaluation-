
from document_chunker import split_documents
from embedding_model import get_embedding_model
from vector_store import create_vector_store
from llm_model import get_llm
from rag_chain import get_rag_answer

from langchain_core.documents import Document


# Step 1: Create sample documents
documents = [
    Document(
        page_content="Machine Learning is a subset of Artificial Intelligence.",
        metadata={"source": "ml.txt"}
    ),
    Document(
        page_content="Deep Learning uses neural networks to learn complex patterns.",
        metadata={"source": "dl.txt"}
    ),
    Document(
        page_content="Python is a popular programming language.",
        metadata={"source": "python.txt"}
    ),
    Document(
        page_content="Nepal is a beautiful country in South Asia.",
        metadata={"source": "nepal.txt"}
    )
]


# Step 2: Split documents into chunks
chunks = split_documents(documents)

print("Total Chunks:", len(chunks))


# Step 3: Load embedding model
embedding_model = get_embedding_model()


# Step 4: Create FAISS vector store
vector_store = create_vector_store(
    chunks,
    embedding_model
)

print("FAISS Vector Store Created!")


# Step 5: Initialize Groq LLM
llm = get_llm()

print("Groq LLM Initialized!")


# Step 6: Ask a question
question = "What is Machine Learning?"


# Step 7: Generate RAG answer
result = get_rag_answer(
    question=question,
    vector_store=vector_store,
    llm=llm,
    k=3
)


# Step 8: Display results
print("\nUser Question:")
print(question)

print("\nRetrieved Documents:")

for doc in result["retrieved_documents"]:
    print("-", doc.page_content)

print("\nFinal AI Answer:")
print(result["answer"])
