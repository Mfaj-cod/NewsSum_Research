import json
from typing import List, Dict

class NewsSummSample:
    def __init__(self, cluster_id: str, documents: List[str], summary: str, metadata: Dict):
        self.cluster_id = cluster_id
        self.documents = documents
        self.summary = summary
        self.metadata = metadata


class NewsSummDataset:
    """
    Unified loader for NewsSumm multi-document clusters.
    Each item returns:
      - cluster_id
      - list of document texts
      - human reference summary
      - metadata (titles, dates, sources, etc.)
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.samples = self._load()

    def _load(self):
        # Expecting a JSONL/JSON file after preprocessing
        if self.data_path.endswith(".json"):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif self.data_path.endswith(".jsonl"):
            data = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            raise ValueError("Unsupported dataset format. Please use JSON or JSONL.")

        samples = []
        for item in data:
            sample = NewsSummSample(
                cluster_id=item["cluster_id"],
                documents=item["documents"],
                summary=item["summary"],
                metadata=item.get("metadata", {})
            )
            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
