# Experiment Tracking & Reproducibility

This project follows a **config-driven, fully reproducible experiment protocol**.  
Any future intern or researcher can:

> Re-run the same experiment and obtain similar results.

---

## 1. Experiment Folder Structure

Every experiment creates a dedicated folder inside:

```bash
results/led_baseline_run_001/
├── config.yaml
├── meta.json
├── summary.json
└── checkpoint/
```


---

## 2. What Is Stored Per Run?

### 2.1 `config.yaml`
A **snapshot of the exact configuration** used for the run:

- Dataset paths
- Model name
- Hyperparameters
- Seed
- Training settings

This guarantees the run can be reproduced exactly.

---

### 2.2 `meta.json`
Stores **system and runtime metadata**:

- Start time
- Device (CPU / GPU)
- Environment information

Example:

```json
{
  "start_time": "2026-01-28 22:15:10",
  "device": "cpu"
}
```

### 2.3 summary.json

Stores final results of the experiment:

- Metrics (loss, ROUGE, BERTScore, etc.)
- Total runtime
- End time

Example:
```bash
{
  "metrics": {
    "loss": 15.82,
    "rouge1": 0.0,
    "rouge2": 0.0,
    "rougeL": 0.0,
    "bertscore": 0.0
  },
  "runtime_seconds": 523.4,
  "end_time": "2026-01-28 22:24:33"
}
```

### 2.4 checkpoint/

If the model is saved, this folder contains:

- Model weights
- Tokenizer files
- Generation config

This allows: Resume training or run inference from the same state.

## 3. How Reproducibility Works

Every experiment is launched using a YAML config file:
```bash
python scripts/train_baseline.py --config configs/led_baseline.yaml
```
That same config is copied into the results folder:
```bash
results/<run_name>/config.yaml
```
To reproduce the experiment:
```bash
python scripts/train_baseline.py --config results/<run_name>/config.yaml
```
---

## 4. What Is Controlled by the Config?

- Random seed
- Model name
- Dataset paths
- Hyperparameters
- Training settings
- Output directory

This ensures: The same experiment can be re-run in the future with minimal variance.
---
## 5. Why This Matters

This system enforces:

- Scientific reproducibility
- Transparent experiment management
- Auditable results
- Clean comparison between runs

## 6. Current Project Usage

In this project:

led_baseline_run_001 demonstrates:

- A fully tracked baseline run
- With config, metadata, summary, and checkpoint

The same system will be used for:

- Novel model experiments
- Future ablations
- Hyperparameter sweeps

## 7. Summary

Every experiment is config-driven, logged, reproducible, and auditable.

This makes the project suitable for:

- Research
- Collaboration
- Benchmarking
- Long-term extension

