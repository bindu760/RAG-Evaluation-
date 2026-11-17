
import sys
from pathlib import Path

# Add src folder to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from langchain_core.documents import Document

from document_chunker import split_documents
from embedding_model import get_embedding_model
from vector_store import create_vector_store
from llm_model import get_llm
from rag_chain import get_rag_answer

from generation_dataset import generation_dataset


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


# 2. Split documents into chunks
chunks = split_documents(documents)


# 3. Initialize embedding model and FAISS
embedding_model = get_embedding_model()

vector_store = create_vector_store(
    chunks,
    embedding_model
)


# 4. Initialize Groq LLM
llm = get_llm()


# 5. Generate answers and collect evaluation data
evaluation_data = []

for item in generation_dataset:

    question = item["question"]

    result = get_rag_answer(
        question=question,
        vector_store=vector_store,
        llm=llm,
        k=3
    )

    contexts = [
        doc.page_content
        for doc in result["retrieved_documents"]
    ]

    evaluation_data.append({
        "question": question,
        "answer": result["answer"],
        "contexts": contexts,
        "ground_truth": item["ground_truth"]
    })

    print("\nQuestion:", question)
    print("Generated Answer:", result["answer"])


# 6. Display collected evaluation data
print("\n========== GENERATION DATA ==========")

for item in evaluation_data:
    print(item)
