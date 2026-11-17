
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)


def get_generation_metrics():
    """
    Return metrics used to evaluate RAG answer generation.
    """

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        AnswerCorrectness(),
    ]

    return metrics
