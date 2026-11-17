
from pathlib import Path
import pandas as pd


# 1. Define project paths
project_root = Path(__file__).resolve().parents[1]

results_dir = project_root / "evaluation" / "results"

retrieval_file = results_dir / "retrieval_evaluation.csv"
generation_file = results_dir / "generation_evaluation.csv"


# 2. Check whether CSV files exist
if not retrieval_file.exists():
    raise FileNotFoundError("Retrieval evaluation CSV not found.")

if not generation_file.exists():
    raise FileNotFoundError("Generation evaluation CSV not found.")


# 3. Load CSV files
retrieval_df = pd.read_csv(retrieval_file)
generation_df = pd.read_csv(generation_file)


# 4. Calculate average retrieval scores
retrieval_scores = retrieval_df[
    ["precision", "recall", "hit_rate", "mrr"]
].mean()


# 5. Calculate average generation scores
generation_scores = generation_df[
    ["faithfulness", "answer_relevancy", "answer_correctness"]
].mean()


# 6. Combine all scores
benchmark_scores = pd.concat([
    retrieval_scores,
    generation_scores
])


# 7. Display combined report
print("\n========== RAG BENCHMARK REPORT ==========")

print("\nRETRIEVAL METRICS")
print(retrieval_scores.round(4))

print("\nGENERATION METRICS")
print(generation_scores.round(4))

print("\nOVERALL BENCHMARK")
print(benchmark_scores.round(4))


# 8. Save combined report
report_file = results_dir / "benchmark_report.csv"

benchmark_scores.rename("average_score").to_csv(
    report_file,
    header=True
)

print(f"\nBenchmark report saved: {report_file}")
