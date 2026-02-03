# Novel Model Specification: Hierarchical Planner–Generator for Multi-Document Summarization

## 1. Motivation

Standard long-context encoder–decoder models such as LED and Long-T5 treat multi-document summarization as a flat sequence-to-sequence problem: all documents are concatenated and fed into a single encoder. This approach suffers from:

- Loss of document-level structure
- Weak global content planning
- Redundancy and topic drifting
- Inefficient usage of long context windows

Human summarization, however, is hierarchical:

1. First decide **what** to say (content planning)
2. Then decide **how** to say it (surface realization)

We propose a **Hierarchical Planner–Generator (HPG)** architecture that explicitly separates **content planning** from **summary generation**.

---

## 2. High-Level Architecture
```bash
Multi-Document Cluster
│
▼
┌──────────────────────────┐
│ Document Encoder │ (shared encoder, per-document embeddings)
└──────────────────────────┘
│
▼
┌──────────────────────────┐
│ Content Planner │ (selects salient topics / segments)
└──────────────────────────┘
│
▼
┌──────────────────────────┐
│ Summary Generator │ (conditional abstractive decoder)
└──────────────────────────┘
│
▼
Final Summary
```

---

## 3. Model Components

### 3.1 Document Encoder

- Each document in the cluster is encoded independently using a **shared Transformer encoder**.
- The output is a set of document-level representations:
D = {d₁, d₂, ..., dₙ}
- This preserves document boundaries and avoids early fusion collapse.

---

### 3.2 Content Planner

The Content Planner is a lightweight transformer or pointer-style network that:

- Attends over document representations
- Selects or weights salient content units (sentences / segments / latent topics)
- Produces a **content plan**:

P = [p₁, p₂, ..., pₖ]


Where each `pᵢ` represents a planned semantic unit or focus region.

This stage explicitly models:

- Coverage
- Ordering
- Salience
- Redundancy avoidance

---

### 3.3 Summary Generator

- A standard encoder–decoder Transformer (e.g., BART / T5 / LED-style decoder)
- Conditioned on:
  - Original document encodings
  - The content plan `P`
- Generates the final abstractive summary autoregressively.

---

## 4. Training Strategy

The model is trained in two phases:

### Phase 1: Warm-Start Generator

- Train the generator as a normal summarization model using: Documents → Summary

### Phase 2: Joint Planning + Generation

- Introduce the planner
- Optimize: Documents → Plan → Summary
- Loss function: L = L_generation + λ * L_planning


Where:
- `L_generation` = standard cross-entropy loss
- `L_planning` = optional auxiliary loss (coverage, ordering, salience)

---

## 5. Inference Procedure

At inference time:

1. Encode all documents
2. Generate a content plan
3. Condition the decoder on the plan
4. Generate the final summary

This enforces:

- Better global coherence
- Better topic coverage
- Reduced redundancy

---

## 6. Why This Improves Over LED / Long-T5

| Aspect | Flat Models (LED) | Proposed HPG |
|-------|-------------------|-------------|
| Structure | Flat concatenation | Hierarchical |
| Planning | Implicit | Explicit |
| Redundancy | High | Reduced |
| Long summaries | Unstable | More coherent |
| Interpretability | Low | High (inspect plan) |

---

## 7. Implementation Plan in This Repository

- `models/novel_model.py` will implement:
  - Document encoder wrapper
  - Planner module
  - Generator wrapper
- `scripts/train_novel.py` will:
  - First warm-start from baseline
  - Then enable planner training

Due to resource constraints, only a **prototype version** is planned during this internship phase.

---

## 8. Expected Research Contribution

- Introduces **explicit planning** into multi-document summarization
- Bridges extractive planning and abstractive generation
- Improves controllability and interpretability of long-form summarization models
- Provides a modular framework for future research extensions

---

## 9. Summary

This project proposes a **Hierarchical Planner–Generator** architecture that decomposes multi-document summarization into:

> **Planning → Realization**

This mirrors how humans write long summaries and addresses key limitations of flat long-context Transformer models.

---

*This document describes the proposed novel model architecture for the NewsSumm benchmark project.*