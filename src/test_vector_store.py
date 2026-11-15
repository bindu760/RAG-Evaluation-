
from langchain_core.documents import Document

from document_chunker import split_documents
from embedding_model import get_embedding_model
from vector_store import create_vector_store, search_documents


# 1. Create sample documents
documents = [
    Document(
        page_content="Python is a popular programming language.",
        metadata={"source": "python.txt"}
    ),
    Document(
        page_content="Machine Learning is a subset of Artificial Intelligence.",
        metadata={"source": "ml.txt"}
    ),
    Document(
        page_content="Deep Learning uses neural networks.",
        metadata={"source": "dl.txt"}
    ),
    Document(
        page_content="Nepal is a beautiful country.",
        metadata={"source": "nepal.txt"}
    )
]


# 2. Split documents into chunks
chunks = split_documents(
    documents,
    chunk_size=100,
    chunk_overlap=20
)

print("Total Chunks:", len(chunks))


# 3. Load embedding model
embeddings = get_embedding_model()


# 4. Create FAISS vector database
vector_store = create_vector_store(
    chunks,
    embeddings
)

print("\nFAISS Vector Store Created Successfully!")


# 5. Search documents
query = "What is Machine Learning?"

results = search_documents(
    vector_store,
    query,
    k=3
)


# 6. Display search results
print("\nSearch Query:", query)

for i, result in enumerate(results):
    print(f"\nResult {i + 1}:")
    print("Content:", result.page_content)
    print("Metadata:", result.metadata)
