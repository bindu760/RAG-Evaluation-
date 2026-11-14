
from document_chunker import split_documents
from langchain_core.documents import Document

# Create sample document
sample_text = """
Artificial Intelligence is transforming the world.
Machine Learning is a subset of Artificial Intelligence.
Deep Learning uses neural networks to learn complex patterns.
RAG combines information retrieval with language generation.
""" * 20

document = Document(
    page_content=sample_text,
    metadata={"source": "sample.txt"}
)

# Split document into chunks
chunks = split_documents(
    [document],
    chunk_size=100,
    chunk_overlap=20
)

# Display results
print("Total Chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}")
    print("Text Length:", len(chunk.page_content))
    print("Content:", chunk.page_content)
    print("Metadata:", chunk.metadata)
