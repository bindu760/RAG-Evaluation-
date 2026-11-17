
from generation_metrics import get_generation_metrics

metrics = get_generation_metrics()

print("Generation Evaluation Metrics:")

for metric in metrics:
    print(metric.name)
