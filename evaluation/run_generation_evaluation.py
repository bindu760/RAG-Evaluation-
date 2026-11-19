
from langchain_core.documents import Document
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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


# 2. Create chunks
chunks = split_documents(documents)


# 3. Initialize embedding model
embedding_model = get_embedding_model()


# 4. Create FAISS vector store
vector_store = create_vector_store(
    chunks,
    embedding_model
)


# 5. Initialize LLM
llm = get_llm()


# 6. Collect generation and performance data
generation_data = []

for item in generation_dataset:

    question = item["question"]
    ground_truth = item["ground_truth"]

    result = get_rag_answer(
        question,
        vector_store,
        llm,
        k=3
    )

    answer = result["answer"]

    contexts = [
        doc.page_content
        for doc in result["retrieved_documents"]
    ]

    performance_data = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "latency": result["latency"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "total_tokens": result["total_tokens"]
    }

    generation_data.append(performance_data)

    print("\nQuestion:", question)
    print("Generated Answer:", answer)
    print("Latency:", round(result["latency"], 3), "seconds")
    print("Input Tokens:", result["input_tokens"])
    print("Output Tokens:", result["output_tokens"])
    print("Total Tokens:", result["total_tokens"])


# 7. Display complete evaluation data
print("\n========== GENERATION & PERFORMANCE DATA ==========")

for data in generation_data:
    print(data)
