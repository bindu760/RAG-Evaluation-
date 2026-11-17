
import sys
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

# Add project folders to Python path
project_root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "evaluation"))

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


# 2. Split documents
chunks = split_documents(documents)


# 3. Create FAISS vector store
embedding_model = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)


# 4. Evaluate retrieval for each question
k = 3
results = []

for item in evaluation_dataset:

    question = item["question"]
    relevant_doc_ids = item["relevant_doc_ids"]

    retrieved_docs = vector_store.similarity_search(
        question,
        k=k
    )

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

    results.append({
        "question": question,
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "mrr": mrr
    })


# 5. Convert results into DataFrame
df = pd.DataFrame(results)


# 6. Create results directory
results_dir = project_root / "evaluation" / "results"
results_dir.mkdir(parents=True, exist_ok=True)


# 7. Save CSV
csv_path = results_dir / "retrieval_evaluation.csv"

df.to_csv(csv_path, index=False)


# 8. Display results
print("\n========== RETRIEVAL EVALUATION RESULTS ==========")
print(df.to_string(index=False))

print("\n========== AVERAGE RETRIEVAL SCORES ==========")
print(df[["precision", "recall", "hit_rate", "mrr"]].mean())

print(f"\nCSV saved successfully: {csv_path}")
