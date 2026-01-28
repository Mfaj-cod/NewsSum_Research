# This script trains a baseline LED model on a single sample from the NewsSumm dataset.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import torch
from scripts.utils import load_config, set_seed, prepare_experiment_folder, finalize_experiment
from models.baseline_led import LEDSummarizer


def main(args):
    config = load_config(args.config)

    # Setup
    set_seed(config["training"]["seed"])
    out_dir = prepare_experiment_folder(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Starting experiment: {config['experiment']['name']}")
    print(f"Device: {device}")
    print(f"Outputs will be saved to: {out_dir}")

    start_time = time.time()

    # Load 1 sample from dataset
    with open(config["data"]["train_file"], "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[0]  # ONLY ONE SAMPLE

    documents = [" ".join(sample["documents"])]
    summaries = [sample["summary"]]

    # Load model
    model = LEDSummarizer(config["model"]["name"], device)

    print("Running ONE training step (CPU-safe)...")

    loss = model.train_step(
        documents,
        summaries,
        config["data"]["max_input_length"],
        config["data"]["max_target_length"],
    )

    print("Loss:", loss.item())

    # Save checkpoint
    ckpt_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(ckpt_dir)

    metrics = {
        "loss": float(loss.item()),
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bertscore": 0.0
    }

    finalize_experiment(out_dir, metrics, start_time)

    print("Baseline LED run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    main(args)
