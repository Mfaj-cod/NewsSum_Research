# This script trains a model on a single sample from the NewsSumm dataset.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import torch
from scripts.utils import load_config, set_seed, prepare_experiment_folder, finalize_experiment
from models.baseline_led import LEDSummarizer
from models.baseline_generic import load_model_and_tokenizer

def main(args):
    config = load_config(args.config)
    portion = args.sample

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

    sample = data[:portion]  # Only one sample for this prototype run
    """Remove the '[:portion]' after data to train on full data."""

    documents = [" ".join(s["documents"]) for s in sample]
    summaries = [s["summary"] for s in sample]

    # Loading model
    model_family = config["model"]["family"]
    if model_family == "led":
        model = LEDSummarizer(config["model"]["name"], device)

    else:
        model, tokenizer = load_model_and_tokenizer(
            config["model"]["name"],
            config["model"]["type"]
        )
        model.to(device)
    

    print("Running ONE training step (CPU-safe)...")

    if model_family == "led":
        # Custom LED wrapper
        loss = model.train_step(
            documents,
            summaries,
            config["data"]["max_input_length"],
            config["data"]["max_target_length"],
        )

    else:
        # Generic HF seq2seq models (LongT5, FLAN-T5, etc.)
        tokenizer = tokenizer  # from generic loader

        inputs = tokenizer(
            documents,
            truncation=True,
            padding=True,
            max_length=config["data"]["max_input_length"],
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        labels = tokenizer(
            text_target=summaries,
            truncation=True,
            padding=True,
            max_length=config["data"]["max_target_length"],
            return_tensors="pt"
        ).input_ids.to(device)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )

        loss = outputs.loss


    print("Loss:", loss.item())

    # Saving checkpoint
    ckpt_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    if model_family == "led":
        # Custom LED wrapper
        model.save(ckpt_dir)
    else:
        # HuggingFace models
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)


    metrics = {
        "loss": float(loss.item()),
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bertscore": 0.0
    }

    finalize_experiment(out_dir, metrics, start_time)

    print(f"config[{config['model']['name']}] run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sample", type=int, required=False, default=1, help="No. of rows to use from dataset")
    args = parser.parse_args()
    main(args)
