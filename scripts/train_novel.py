import os
# import json
# import argparse
import sys
# import yaml
# import time
# from datetime import datetime

# import torch
# from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from models.novel_model import HierarchicalPlannerGenerator


# def load_config(path):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def load_dataset(path, sample=None):
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     if sample is not None:
#         data = data[:sample]

#     return data


# def main(args):
#     config = load_config(args.config)

#     run_name = config["experiment"]["name"]
#     out_dir = os.path.join("results", run_name)
#     os.makedirs(out_dir, exist_ok=True)

#     print(f"Starting Novel Model experiment: {run_name}")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)
#     print("Outputs will be saved to:", out_dir)

#     # Load dataset
#     dataset = load_dataset(
#         config["data"]["train_file"],
#         sample=args.sample
#     )

#     # Extract documents and summaries
#     documents = [" ".join(ex["documents"]) for ex in dataset]
#     summaries = [ex["summary"] for ex in dataset]

#     # Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         config["model"]["base_model"],
#         use_fast=True
#     )

#     planner_hidden_size = config["model"].get("planner_hidden_size", None)
#     # Model
#     model = HierarchicalPlannerGenerator(
#         base_model_name=config["model"]["base_model"],
#         planner_hidden_size=planner_hidden_size
#     ).to(device)

#     model.train()

#     # Tokenization (REAL DATA)
#     inputs = tokenizer(
#         documents,
#         truncation=True,
#         padding=True,
#         max_length=config["data"]["max_input_length"],
#         return_tensors="pt"
#     )

#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     labels = tokenizer(
#         text_target=summaries,
#         truncation=True,
#         padding=True,
#         max_length=config["data"]["max_target_length"],
#         return_tensors="pt"
#     ).input_ids.to(device)

#     # ONE prototype training step
#     print("Running ONE prototype training step for Novel Model...")

#     outputs = model(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         labels=labels
#     )

#     loss = outputs["loss"]
#     print("Loss:", loss.item())

#     # Save checkpoint
#     ckpt_dir = os.path.join(out_dir, "checkpoint")
#     os.makedirs(ckpt_dir, exist_ok=True)

#     # Save generator.model + tokenizer for evaluation compatibility
#     model.generator.model.save_pretrained(ckpt_dir)
#     tokenizer.save_pretrained(ckpt_dir)

#     # Save metadata
#     summary = {
#         "metrics": {
#             "loss": loss.item(),
#             "rouge1": 0.0,
#             "rouge2": 0.0,
#             "rougeL": 0.0,
#             "bertscore": 0.0
#         },
#         "runtime_seconds": 0.0,
#         "end_time": datetime.now().isoformat()
#     }

#     with open(os.path.join(out_dir, "summary.json"), "w") as f:
#         json.dump(summary, f, indent=2)

#     with open(os.path.join(out_dir, "config.yaml"), "w") as f:
#         yaml.dump(config, f)

#     with open(os.path.join(out_dir, "meta.json"), "w") as f:
#         json.dump({
#             "model": "Hierarchical Planner-Generator (HPG)",
#             "base_model": config["model"]["base_model"],
#             "samples_used": len(dataset),
#             "device": str(device)
#         }, f, indent=2)

#     print("Novel model prototype run complete.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True)
#     parser.add_argument("--sample", type=int, default=None, help="No. of rows to use from dataset")
#     args = parser.parse_args()

#     main(args)






import os
import json
import yaml
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW
from tqdm import tqdm

from models.novel_model import HierarchicalPlannerGenerator


# Dataset
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


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)

    run_name = config["experiment"]["name"]
    out_dir = os.path.join("results", run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    fp16 = config["training"].get("fp16", False)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    model = HierarchicalPlannerGenerator(
        base_model_name=config["model"]["base_model"]
    ).to(device)

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

    print("Training Novel model...")

    for step, batch in enumerate(tqdm(loader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=fp16):
            out = model(**batch)
            loss = out["loss"] / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

    ckpt_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.generator.model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    print("Novel training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    main(args)
