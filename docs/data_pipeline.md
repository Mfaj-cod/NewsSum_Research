# NewsSumm Data Pipeline

## 1. Raw Data

The NewsSumm dataset consists of multi-document news clusters, where each cluster contains:
- Multiple related news articles
- One human-written abstractive summary
- Metadata (source, date, title, etc.)

For development and demonstration, we currently support an XLSX/JSON-based input format.

---

## 2. Preprocessing Steps

Implemented in: `scripts/preprocess.py`

Steps:
1. Load raw data (XLSX or JSON)
2. Remove HTML and boilerplate using BeautifulSoup
3. Normalize whitespace
4. Convert into a unified JSON format:

```json
{
  "cluster_id": "...",
  "documents": ["doc1", "doc2", "..."],
  "summary": "...",
  "metadata": {...}
}
```

---

## 3. Dataset Loader
...
Implemented in: newssumm_dataset.py

- The loader exposes each sample as:
- cluster_id
- list of document texts
- reference summary
- metadata

This format is compatible with both encoder-decoder and decoder-only models.

### 4. Statistics

Implemented in: scripts/compute_stats.py

Computes:

- Average documents per cluster
- Average document length (words)
- Average summary length (words)

This helps in deciding:

- Max input length
- Truncation strategies
- Model context window requirements

---

# Step 5 — Run It

Run:

```bash
python scripts/preprocess.py --input data/NewsSumm_Dataset.xlsx --output data/newssumm_processed
```
Then:
```bash
python scripts/compute_stats.py --data data/newssumm_processed/newssumm_processed.json
```