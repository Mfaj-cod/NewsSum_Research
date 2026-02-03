# Baselines
```bash
python scripts/train_baseline.py --config configs/led_baseline.yaml --sample 50
python scripts/run_evaluation.py --run_dir results/led_baseline_run_001 --data data/newssumm_processed/newssumm_processed.json --sample 50

python scripts/train_baseline.py --config configs/longt5.yaml --sample 50
python scripts/run_evaluation.py --run_dir results/longt5_run_001 --data data/newssumm_processed/newssumm_processed.json --sample 50
```
# Novel model
```bash
python scripts/train_novel.py --config configs/novel_model.yaml --sample 50
python scripts/run_evaluation.py --run_dir results/novel_model_run_001 --data data/newssumm_processed/newssumm_processed.json --sample 50
```