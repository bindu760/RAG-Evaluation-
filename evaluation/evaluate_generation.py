
import sys
from pathlib import Path

from datasets import Dataset
from ragas import evaluate

# Add project folders to Python path
project_root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "evaluation"))

from generation_metrics import get_generation_metrics
from run_generation_evaluation import evaluation_data
from llm_model import get_llm
from embedding_model import get_embedding_model


# 1. Convert evaluation data into Hugging Face Dataset
dataset = Dataset.from_list(evaluation_data)


# 2. Initialize LLM and embedding model
llm = get_llm()
embeddings = get_embedding_model()


# 3. Get generation metrics
metrics = get_generation_metrics()


# 4. Run Ragas evaluation
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings
)


# 5. Display evaluation scores
print("\n========== RAGAS GENERATION EVALUATION ==========")

print(results)
