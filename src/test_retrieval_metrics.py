
from langchain_core.documents import Document

from retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank
)


# Sample retrieved documents
retrieved_docs = [
    Document(
        page_content="Machine Learning is a subset of AI.",
        metadata={"source": "ml.txt"}
    ),
    Document(
        page_content="Deep Learning uses neural networks.",
        metadata={"source": "dl.txt"}
    ),
    Document(
        page_content="Python is a programming language.",
        metadata={"source": "python.txt"}
    )
]


# Ground truth: relevant document IDs
relevant_doc_ids = {"ml.txt"}


# Calculate metrics
precision = precision_at_k(retrieved_docs, relevant_doc_ids, k=3)
recall = recall_at_k(retrieved_docs, relevant_doc_ids, k=3)
hit_rate = hit_rate_at_k(retrieved_docs, relevant_doc_ids, k=3)
mrr = mean_reciprocal_rank(retrieved_docs, relevant_doc_ids, k=3)


# Display results
print("Retrieval Evaluation Results")
print("----------------------------")
print("Precision@3:", precision)
print("Recall@3:", recall)
print("Hit Rate@3:", hit_rate)
print("MRR:", mrr)
