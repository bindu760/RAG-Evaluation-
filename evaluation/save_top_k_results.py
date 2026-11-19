
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from pathlib import Path

from langchain_core.documents import Document

from src.document_chunker import split_documents
from src.embedding_model import get_embedding_model
from src.vector_store import create_vector_store
from evaluation.evaluation_dataset import evaluation_dataset
from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank
)


# 1. Create sample documents
documents = [
    Document(
        page_content="Machine Learning is a subset of Artificial Intelligence.",
        metadata={"source": "ml.txt"}
    ),
    Document(
        page_content="Deep Learning uses neural networks to learn patterns from data.",
        metadata={"source": "dl.txt"}
    ),
    Document(
        page_content="Python is a popular programming language.",
        metadata={"source": "python.txt"}
    ),
    Document(
        page_content="Nepal is located in South Asia.",
        metadata={"source": "nepal.txt"}
    )
]


# 2. Create chunks and vector store
chunks = split_documents(documents)

embedding_model = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)


# 3. Compare different Top-K values
top_k_values = [3, 5, 10]

results = []

for k in top_k_values:

    precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    mrr_scores = []

    for item in evaluation_dataset:

        question = item["question"]
        relevant_doc_ids = item["relevant_doc_ids"]

        retrieved_docs = vector_store.similarity_search(
            question,
            k=k
        )

        precision_scores.append(
            precision_at_k(retrieved_docs, relevant_doc_ids, k)
        )

        recall_scores.append(
            recall_at_k(retrieved_docs, relevant_doc_ids, k)
        )

        hit_rate_scores.append(
            hit_rate_at_k(retrieved_docs, relevant_doc_ids, k)
        )

        mrr_scores.append(
            mean_reciprocal_rank(retrieved_docs, relevant_doc_ids, k)
        )

    results.append({
        "top_k": k,
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
        "hit_rate": sum(hit_rate_scores) / len(hit_rate_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores)
    })


# 4. Save results to CSV
results_df = pd.DataFrame(results)

results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output_file = results_dir / "top_k_comparison.csv"

results_df.to_csv(output_file, index=False)

print("\nTOP-K COMPARISON RESULTS")
print(results_df.to_string(index=False))

print(f"\nResults saved to: {output_file}")
