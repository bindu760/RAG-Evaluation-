
def precision_at_k(retrieved_docs, relevant_doc_ids, k=3):
    """
    Calculate Precision@K.
    Measures how many retrieved documents are relevant.
    """

    top_k_docs = retrieved_docs[:k]

    if not top_k_docs:
        return 0.0

    relevant_count = sum(
        1 for doc in top_k_docs
        if doc.metadata.get("source") in relevant_doc_ids
    )

    return relevant_count / len(top_k_docs)


def recall_at_k(retrieved_docs, relevant_doc_ids, k=3):
    """
    Calculate Recall@K.
    Measures how many relevant documents were retrieved.
    """

    if not relevant_doc_ids:
        return 0.0

    top_k_docs = retrieved_docs[:k]

    relevant_count = sum(
        1 for doc in top_k_docs
        if doc.metadata.get("source") in relevant_doc_ids
    )

    return relevant_count / len(relevant_doc_ids)


def hit_rate_at_k(retrieved_docs, relevant_doc_ids, k=3):
    """
    Checks whether at least one relevant document was retrieved.
    """

    top_k_docs = retrieved_docs[:k]

    for doc in top_k_docs:
        if doc.metadata.get("source") in relevant_doc_ids:
            return 1.0

    return 0.0


def mean_reciprocal_rank(retrieved_docs, relevant_doc_ids, k=3):
    """
    Measures the ranking position of the first relevant document.
    """

    top_k_docs = retrieved_docs[:k]

    for rank, doc in enumerate(top_k_docs, start=1):
        if doc.metadata.get("source") in relevant_doc_ids:
            return 1 / rank

    return 0.0
