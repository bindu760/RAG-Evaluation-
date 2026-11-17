
import sys
from pathlib import Path

import pandas as pd
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


# 1. Convert evaluation data into Dataset
dataset = Dataset.from_list(evaluation_data)


# 2. Initialize models
llm = get_llm()
embeddings = get_embedding_model()


# 3. Get metrics
metrics = get_generation_metrics()


# 4. Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings
)


# 5. Convert individual evaluation scores to DataFrame
scores = results.to_pandas()

# Include original questions and generated answers
input_data = pd.DataFrame(evaluation_data)

final_results = pd.concat(
    [
        input_data[["question", "answer", "ground_truth"]],
        scores[["faithfulness", "answer_relevancy", "answer_correctness"]]
    ],
    axis=1
)


# 6. Create results directory
results_dir = project_root / "evaluation" / "results"
results_dir.mkdir(parents=True, exist_ok=True)


# 7. Save CSV
csv_path = results_dir / "generation_evaluation.csv"

final_results.to_csv(csv_path, index=False)


print("\n========== SAVED EVALUATION RESULTS ==========")
print(final_results)

print(f"\nCSV saved successfully: {csv_path}")
