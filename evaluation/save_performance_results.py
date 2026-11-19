
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.document_chunker import split_documents
from src.embedding_model import get_embedding_model
from src.vector_store import create_vector_store
from src.llm_model import get_llm
from src.rag_chain import get_rag_answer

from evaluation.generation_dataset import generation_dataset


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


# 2. Create vector store
chunks = split_documents(documents)

embedding_model = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)

# 3. Initialize LLM
llm = get_llm()


# 4. Collect performance results
performance_results = []

for item in generation_dataset:

    question = item["question"]

    result = get_rag_answer(
        question,
        vector_store,
        llm,
        k=3
    )

    performance_results.append({
        "question": question,
        "latency": result["latency"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "total_tokens": result["total_tokens"]
    })


# 5. Convert results into DataFrame
performance_df = pd.DataFrame(performance_results)


# 6. Calculate average performance
average_row = {
    "question": "AVERAGE",
    "latency": performance_df["latency"].mean(),
    "input_tokens": performance_df["input_tokens"].mean(),
    "output_tokens": performance_df["output_tokens"].mean(),
    "total_tokens": performance_df["total_tokens"].mean()
}

performance_df = pd.concat(
    [performance_df, pd.DataFrame([average_row])],
    ignore_index=True
)


# 7. Save CSV file
results_dir = PROJECT_ROOT / "evaluation" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output_file = results_dir / "performance_evaluation.csv"

performance_df.to_csv(output_file, index=False)


# 8. Display results
print("\n========== PERFORMANCE EVALUATION ==========")

print(performance_df.to_string(index=False))

print(f"\nResults saved to: {output_file}")
