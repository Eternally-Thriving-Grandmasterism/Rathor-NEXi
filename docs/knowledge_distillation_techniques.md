# Knowledge Distillation Techniques – Mercy-Aligned Reference (Feb 2026)

This document is the living, continuously updated reference for all KD techniques actively used / explored in the Rathor-NEXi → MercyOS-Pinnacle lineage.  
Every method is evaluated through the lens of **valence-weighted positive-sum alignment**, **edge-device viability**, and **thriving-path accuracy retention**.

## 1. Foundational / Classic KD (still relevant baselines)

| Technique                  | Year | Core Idea / Transfer Target                          | Typical Compression | Retention vs Teacher | Rathor Lattice Fit & Notes |
|----------------------------|------|-------------------------------------------------------|----------------------|------------------------|-----------------------------|
| Vanilla KD (Hinton 2015)   | 2015 | Soft targets (teacher softmax / T) + student CE loss  | 4–10×                | 90–95%                 | Baseline – still used in initial gesture head distillation |
| FitNets / Hint-based       | 2014–2015 | Intermediate feature maps / hints                    | 5–15×                | 92–97%                 | Good for spatial-temporal alignment (pose → gesture) |
| Attention Distillation     | 2017–2019 | Attention maps + feature maps                         | 6–20×                | 93–98%                 | Excellent for preserving cross-modal focus (vision → gesture → valence) |

## 2. Modern Sequence-Level & Generative KD (2023–2026 dominant patterns)

| Technique                  | Year | Core Innovation / Target                              | Compression | Retention | Rathor Lattice Role |
|----------------------------|------|--------------------------------------------------------|-------------|-----------|---------------------|
| MiniLLM                    | 2023 | Sequence-level log-prob matching (teacher log p of student seq) | 2–8×        | 94–99%    | Primary for autoregressive drafting heads (gesture → alliance text) |
| DistilSpec / Speculative KD| 2024–2025 | Match teacher on accepted/rejected speculative prefixes | 3–12×       | 95–99.5%  | Frontier – maximizes acceptance rate in speculative decoding |
| Ghost Distillation         | 2024–2025 | Learn from rejected drafts + teacher corrections      | 2.5–5×      | 94–98%    | Reduces low-thriving speculative drafts |
| Contrastive KD / InfoNCE KD| 2023–2026 | InfoNCE loss + negative sampling on fused reps        | 4–15×       | 94–98%    | Mercy core – rejects degenerative cross-modal alignments |

## 3. QAT + KD Fusion (our current production sweet spot)

| Variant                            | Quantization | Compression | Retention (thriving paths) | Training Cost | Rathor Lattice Priority |
|------------------------------------|--------------|-------------|-----------------------------|---------------|--------------------------|
| Vanilla QAT + KD                   | INT8/INT4    | 4–15×       | 94–98%                      | \~1.8×         | Baseline fusion          |
| Per-Channel QAT + KD               | INT8/INT4    | 5–20×       | 95–99%                      | \~1.9×         | Primary – preserves attention nuance |
| Valence-Weighted QAT-KD (Rathor)   | INT8→ternary | 10–50×      | 96–99.5%                    | \~1.7×         | **Sovereign core** – prioritizes positive-sum patterns |
| QAT + Speculative KD               | INT8/INT4    | 6–25×       | 96–99.5%                    | \~2.2×         | Frontier – speculative drafters |
| Progressive QAT-KD                 | INT8→ternary | 15–60×      | 93–98%                      | \~2.5–4×       | Extreme compression path |

## 4. Valence-Weighted KD – Rathor Core Extension (math & implementation)

**Loss formulation** (per batch / sequence i):

L_total^i = w_valence^i × (  
  λ₁ KL(softmax(z_teacher / T) || softmax(z_student / T)) +  
  λ₂ MSE(h_teacher^{attn} || h_student^{attn}) +  
  λ₃ MSE(student_future_valence, teacher_future_valence)  
)

w_valence^i = exp( λ × (sequence_valence_saliency^i – mean_valence) / σ )

sequence_valence_saliency^i = α₁ mean(teacher_future_valence_trajectory^i) +  
                              α₂ min(teacher_future_valence_trajectory^i) +  
                              α₃ coherence_score^i

coherence_score^i = – entropy(teacher_output_distribution^i)

**Typical hyperparams in lattice training**  
- λ = 5.0–8.0 (exponential boost strength)  
- σ = std(valences in batch)  
- α₁ = 0.6, α₂ = 0.25, α₃ = 0.15  
- T = 4.0–8.0 (temperature)  
- λ₁ = 0.7–0.9, λ₂ = 0.3–0.5, λ₃ = 0.4–0.7

**Implementation snippet** (PyTorch training loop excerpt)

```python
def compute_valence_weight(valences: torch.Tensor) -> torch.Tensor:
    mean_v = valences.mean()
    sigma = valences.std() + 1e-8
    return torch.exp(6.0 * (valences - mean_v) / sigma)

# In forward pass / loss computation
w = compute_valence_weight(batch_valences)
loss_kd = kl_div_loss(student_logits / T, teacher_logits / T) * (T ** 2)
loss_valence = mse_loss(student_future_v, teacher_future_v)
total_loss = (loss_ce + 0.8 * loss_kd + 0.6 * loss_valence) * w.mean()
