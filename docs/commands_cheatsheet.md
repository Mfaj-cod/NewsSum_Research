# LED baseline (safe CPU)
```bash
python scripts/train_baseline.py --config configs/led_baseline.yaml --sample 1
```
# LongT5 baseline
```bash
python scripts/train_baseline.py --config configs/longt5.yaml --sample 1
```
# FLAN-T5
```bash
python scripts/train_baseline.py --config configs/flan_t5_xl.yaml --sample 1
```
# Novel model prototype
```bash
python scripts/train_novel.py --config configs/novel_model.yaml --sample 1
```
# Evaluate any run
```bash
python scripts/evaluate.py --run results/<run_name>
```