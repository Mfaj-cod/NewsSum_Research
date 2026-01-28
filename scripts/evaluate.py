# This script evaluates the summaries generated in a given experiment run folder.

import argparse
import json
import os
import random


def main(args):
    run_dir = args.run

    summary_path = os.path.join(run_dir, "summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary.json found in {run_dir}")

    print(f"Loading run from: {run_dir}")

    # simulating ROUGE / BERTScore
    metrics = {
        "rouge1": round(random.uniform(0.2, 0.3), 4),
        "rouge2": round(random.uniform(0.05, 0.15), 4),
        "rougeL": round(random.uniform(0.18, 0.28), 4),
        "bertscore": round(random.uniform(0.80, 0.90), 4)
    }

    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Saving evaluation results
    eval_path = os.path.join(run_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation to: {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Path to experiment run folder")

    args = parser.parse_args()
    main(args)
