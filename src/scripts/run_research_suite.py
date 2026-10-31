import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_suite import run_suite

if __name__ == "__main__":
    result = run_suite(root="research_datasets", results_dir="results/research_suite", repeats=5, folds=5)
    print("Research suite completed.")
    print(result["run_summary"].to_string(index=False))
    print("\nRouting:")
    print(result["routing"].to_string(index=False))
    print("\nResults saved under results/research_suite/")
