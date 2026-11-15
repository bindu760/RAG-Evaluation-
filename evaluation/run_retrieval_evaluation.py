
import sys
from pathlib import Path

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from langchain_core.documents import Document

from document_chunker import split_documents
from embedding_model import get_embedding_model
from vector_store import create_vector_store

from evaluation_dataset import evaluation_dataset

from retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank
)


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
        page_content="Nepal is a beautiful country located in South Asia.",
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

print("FAISS Vector Store Created Successfully!")


# Step 5: Configure retrieval
k = 3

precision_scores = []
recall_scores = []
hit_rate_scores = []
mrr_scores = []


# Step 6: Evaluate each question
for item in evaluation_dataset:

    question = item["question"]

    relevant_doc_ids = set(item["relevant_doc_ids"])

    # Retrieve documents from FAISS
    retrieved_docs = vector_store.similarity_search(
        question,
        k=k
    )

    # Calculate metrics
    precision = precision_at_k(
        retrieved_docs, relevant_doc_ids, k
    )

    recall = recall_at_k(
        retrieved_docs, relevant_doc_ids, k
    )

    hit_rate = hit_rate_at_k(
        retrieved_docs, relevant_doc_ids, k
    )

    mrr = mean_reciprocal_rank(
        retrieved_docs, relevant_doc_ids, k
    )

    # Store metric scores
    precision_scores.append(precision)
    recall_scores.append(recall)
    hit_rate_scores.append(hit_rate)
    mrr_scores.append(mrr)

    # Display individual results
    print("\nQuestion:", question)

    print("Retrieved Documents:")

    for doc in retrieved_docs:
        print("-", doc.metadata.get("source"))

    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("Hit Rate:", hit_rate)
    print("MRR:", round(mrr, 3))


# Step 7: Calculate average metrics
total_questions = len(evaluation_dataset)

if total_questions == 0:
    raise ValueError("Evaluation dataset is empty.")

print("\n========== FINAL EVALUATION RESULTS ==========")

print("Total Questions:", total_questions)
print("Top K:", k)

print(
    "Average Precision:",
    round(sum(precision_scores) / total_questions, 3)
)

print(
    "Average Recall:",
    round(sum(recall_scores) / total_questions, 3)
)

print(
    "Average Hit Rate:",
    round(sum(hit_rate_scores) / total_questions, 3)
)

print(
    "Average MRR:",
    round(sum(mrr_scores) / total_questions, 3)
)
