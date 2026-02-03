# # This script trains a model on a single sample from the NewsSumm dataset.

# import sys
# import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import argparse
# import time
# import json
# import torch
# from scripts.utils import load_config, set_seed, prepare_experiment_folder, finalize_experiment
# from models.baseline_led import LEDSummarizer
# from models.baseline_generic import load_model_and_tokenizer

# def main(args):
#     config = load_config(args.config)
#     portion = args.sample

#     set_seed(config["training"]["seed"])
#     out_dir = prepare_experiment_folder(config)

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     print(f"Starting experiment: {config['experiment']['name']}")
#     print(f"Device: {device}")
#     print(f"Outputs will be saved to: {out_dir}")

#     start_time = time.time()

#     # Load 1 sample from dataset
#     with open(config["data"]["train_file"], "r", encoding="utf-8") as f:
#         data = json.load(f)

#     sample = data[:portion]  # Only one sample for this prototype run
#     """Remove the '[:portion]' after data to train on full data."""

#     documents = [" ".join(s["documents"]) for s in sample]
#     summaries = [s["summary"] for s in sample]

#     # Loading model
#     model_family = config["model"]["family"]
#     if model_family == "led":
#         model = LEDSummarizer(config["model"]["name"], device)

#     else:
#         model, tokenizer = load_model_and_tokenizer(
#             config["model"]["name"],
#             config["model"]["type"]
#         )
#         model.to(device)
    

#     print("Running ONE training step (CPU-safe)...")

#     if model_family == "led":
#         # Custom LED wrapper
#         loss = model.train_step(
#             documents,
#             summaries,
#             config["data"]["max_input_length"],
#             config["data"]["max_target_length"],
#         )

#     else:
#         # Generic HF seq2seq models (LongT5, FLAN-T5, etc.)
#         tokenizer = tokenizer  # from generic loader

#         inputs = tokenizer(
#             documents,
#             truncation=True,
#             padding=True,
#             max_length=config["data"]["max_input_length"],
#             return_tensors="pt"
#         )

#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         labels = tokenizer(
#             text_target=summaries,
#             truncation=True,
#             padding=True,
#             max_length=config["data"]["max_target_length"],
#             return_tensors="pt"
#         ).input_ids.to(device)

#         outputs = model(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             labels=labels
#         )

#         loss = outputs.loss


#     print("Loss:", loss.item())

#     # Saving checkpoint
#     ckpt_dir = os.path.join(out_dir, "checkpoint")
#     os.makedirs(ckpt_dir, exist_ok=True)

#     if model_family == "led":
#         model.save(ckpt_dir)
#     else:
#         model.save_pretrained(ckpt_dir)
#         tokenizer.save_pretrained(ckpt_dir)
        
#     metrics = {
#         "loss": float(loss.item()),
#         "rouge1": 0.0,
#         "rouge2": 0.0,
#         "rougeL": 0.0,
#         "bertscore": 0.0
#     }

#     finalize_experiment(out_dir, metrics, start_time)

#     print(f"config[{config['model']['name']}] run complete.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--sample", type=int, required=False, default=1, help="No. of rows to use from dataset")
#     args = parser.parse_args()
#     main(args)





import os
import json
import yaml
import argparse
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from tqdm import tqdm


# =========================
# Dataset (no external deps)
# =========================
class NewsSumDataset(Dataset):
    def __init__(self, path, tokenizer, max_in, max_out, sample=None):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if sample:
            data = data[:sample]

        self.docs = [" ".join(x["documents"]) for x in data]
        self.sums = [x["summary"] for x in data]

        self.tokenizer = tokenizer
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        x = self.tokenizer(
            self.docs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_in,
            return_tensors="pt"
        )

        y = self.tokenizer(
            text_target=self.sums[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_out,
            return_tensors="pt"
        )

        item = {
            "input_ids": x["input_ids"].squeeze(0),
            "attention_mask": x["attention_mask"].squeeze(0),
            "labels": y["input_ids"].squeeze(0)
        }

        item["labels"][item["labels"] == self.tokenizer.pad_token_id] = -100
        return item


# =========================
# Utils
# =========================
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# =========================
# Training
# =========================
def main(args):
    config = load_config(args.config)

    run_name = config["experiment"]["name"]
    out_dir = os.path.join("results", run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Saving to:", out_dir)

    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    fp16 = config["training"].get("fp16", False)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["name"]).to(device)

    dataset = NewsSumDataset(
        config["data"]["train_file"],
        tokenizer,
        config["data"]["max_input_length"],
        config["data"]["max_target_length"],
        sample=args.sample
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=3e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    model.train()
    optimizer.zero_grad()

    print("Training...")

    for step, batch in enumerate(tqdm(loader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=fp16):
            out = model(**batch)
            loss = out.loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    # Save checkpoint
    ckpt = os.path.join(out_dir, "checkpoint")
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"loss": float(loss)}, f)

    print("Baseline training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    main(args)
