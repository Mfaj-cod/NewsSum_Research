# This script defines a novel hierarchical planner-generator model for abstractive summarization.
import torch.nn as nn
from transformers import AutoModel, AutoModelForSeq2SeqLM


class DocumentEncoder(nn.Module):
    """
    Shared encoder that encodes each document independently.
    """

    def __init__(self, base_model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Return last hidden state
        return outputs.last_hidden_state


class ContentPlanner(nn.Module):
    """
    Lightweight planner that attends over document representations
    and produces a plan representation.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, doc_embeddings):
        """
        doc_embeddings: (B, N, H) where
            B = batch size
            N = number of documents (or segments)
            H = hidden size
        """
        attn_out, _ = self.attention(doc_embeddings, doc_embeddings, doc_embeddings)
        plan = self.linear(attn_out)
        return plan


class SummaryGenerator(nn.Module):
    """
    Abstractive generator conditioned on document encodings and plan.
    """

    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs


class HierarchicalPlannerGenerator(nn.Module):
    def __init__(self, base_model_name: str, planner_hidden_size: int = None):
        super().__init__()

        self.base_model_name = base_model_name
        self.document_encoder = DocumentEncoder(base_model_name)

        # Get true hidden size from model config (e.g., 768 for LED)
        hidden_size = self.document_encoder.encoder.config.hidden_size

        print(f"[HPG] Using hidden size from backbone: {hidden_size}")

        # Planner always matches backbone dimension
        self.planner = ContentPlanner(hidden_size)

        self.generator = SummaryGenerator(base_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        For now, this is a PROTOTYPE forward pass.

        input_ids: (B, T)
        attention_mask: (B, T)
        labels: (B, T)
        """

        # 1. Encoding documents (prototype: treat as one big document)
        doc_repr = self.document_encoder(input_ids, attention_mask)
        # doc_repr: (B, T, H)

        # 2. Pooling to get a single vector per document (prototype)
        doc_repr_pooled = doc_repr.mean(dim=1, keepdim=True)  # (B, 1, H)

        # 3. Planning over document representations
        plan = self.planner(doc_repr_pooled)  # (B, 1, H)

        # 4. For prototype: ignoring plan injection and just call generator
        # (In full version, plan would be injected via cross-attention or prefix tokens)

        outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "plan": plan
        }

    def generate(self, input_ids, attention_mask, **gen_kwargs):
        """
        Prototype generation method.
        """
        return self.generator.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
