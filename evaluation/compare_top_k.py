
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


# Step 2: Create FAISS vector store
chunks = split_documents(documents)

embedding_model = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)

print("FAISS Vector Store Created Successfully!")


# Step 3: Define Top-K configurations
k_values = [3, 5, 10]

results = []


# Step 4: Evaluate each Top-K configuration
for k in k_values:

    precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    mrr_scores = []

    for item in evaluation_dataset:

        question = item["question"]
        relevant_doc_ids = set(item["relevant_doc_ids"])

        # Retrieve documents
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

        precision_scores.append(precision)
        recall_scores.append(recall)
        hit_rate_scores.append(hit_rate)
        mrr_scores.append(mrr)

    # Calculate average scores
    total = len(evaluation_dataset)

    results.append({
        "top_k": k,
        "precision": sum(precision_scores) / total,
        "recall": sum(recall_scores) / total,
        "hit_rate": sum(hit_rate_scores) / total,
        "mrr": sum(mrr_scores) / total
    })


# Step 5: Display comparison
print("\n========== TOP-K COMPARISON ==========")

print(
    f"{'Top-K':<10}"
    f"{'Precision':<15}"
    f"{'Recall':<12}"
    f"{'Hit Rate':<12}"
    f"{'MRR':<10}"
)

for result in results:

    print(
        f"{result['top_k']:<10}"
        f"{result['precision']:<15.3f}"
        f"{result['recall']:<12.3f}"
        f"{result['hit_rate']:<12.3f}"
        f"{result['mrr']:<10.3f}"
    )
