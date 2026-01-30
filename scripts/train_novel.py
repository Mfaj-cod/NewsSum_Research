# This script trains a baseline LED model on a single sample from the NewsSumm dataset.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import torch

from scripts.utils import load_config, set_seed, prepare_experiment_folder, finalize_experiment
from models.novel_model import HierarchicalPlannerGenerator


def main(args):
    config = load_config(args.config)
    portion = args.sample

    set_seed(config["training"]["seed"])
    out_dir = prepare_experiment_folder(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting Novel Model experiment: {config['experiment']['name']}")
    print(f"Device: {device}")
    print(f"Outputs will be saved to: {out_dir}")

    start_time = time.time()

    # Load one sample from dataset
    with open(config["data"]["train_file"], "r", encoding="utf-8") as f:
        data = json.load(f)


    sample = data[:portion] # Inputed sample for this prototype run
    """Remove the '[:portion]' after data to train on full data."""

    # Fake tokenized inputs (prototype)
    # In real training, this would come from a tokenizer
    input_ids = torch.randint(0, 1000, (1, 128)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = torch.randint(0, 1000, (1, 64)).to(device)

    # Loading model
    model = HierarchicalPlannerGenerator(
        base_model_name=config["model"]["base_model"]
    ).to(device)

    print("Running ONE prototype training step for Novel Model...")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs["loss"]

    print("Loss:", loss.item())

    metrics = {
        "loss": float(loss.item()),
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bertscore": 0.0
    }

    finalize_experiment(out_dir, metrics, start_time)

    print("Novel model prototype run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sample", type=int, required=False, default=1, help="No. of rows to use from dataset")

    args = parser.parse_args()
    main(args)
