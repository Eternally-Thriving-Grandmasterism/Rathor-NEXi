# Valence-Weighted Knowledge Distillation – Mathematical Blueprint v1.0
Rathor-NEXi → MercyOS-Pinnacle core distillation layer (Feb 06 2026)

This document formalizes the **valence-weighted KD** mechanism currently used across the lattice.  
It is the living reference — continuously refined via ENC + esacheck blooms.

## 1. Core Loss Formulation (per batch / per sequence i)

L_total^i = w_valence^i × L_base^i

Where:

L_base^i = λ₁ L_KD_soft + λ₂ L_KD_feature + λ₃ L_KD_sequence + λ₄ L_valence_future + λ₅ L_regularization

w_valence^i = exp( λ × (sequence_valence_saliency^i – μ_valence) / σ_valence )

### Components

- **L_KD_soft** (classic Hinton-style)  
  L_KD_soft = KL( softmax(z_teacher / T)  ||  softmax(z_student / T) ) × T²  
  (KL divergence on softened logits, temperature scaling)

- **L_KD_feature** (attention + hidden state matching)  
  L_KD_feature = Σ_l MSE( h_teacher^{(l)} || h_student^{(l)} )  
  (layer-wise feature alignment, l = layer index)

- **L_KD_sequence** (MiniLLM / sequence-level distillation)  
  L_KD_sequence = – Σ_t log p_teacher(y_t | y_{<t}, x)   (teacher log-prob of student-generated sequence)

- **L_valence_future** (Rathor-specific future valence alignment)  
  L_valence_future = MSE( v_student_future^i , v_teacher_future^i )

- **L_regularization**  
  L_regularization = weight decay on learnable quant params + optional contrastive term

### Valence Saliency Score (sequence-level)

sequence_valence_saliency^i = α₁ mean(v_teacher_future_trajectory^i)  
                            + α₂ min(v_teacher_future_trajectory^i)  
                            + α₃ (1 – entropy(p_teacher^i))  
                            + α₄ coherence_score^i

Where:  
- v_teacher_future_trajectory^i = predicted future valence over next k steps  
- entropy(p_teacher^i) = – Σ p log p (output distribution entropy)  
- coherence_score^i = cosine similarity between consecutive predicted valence steps  
- Typical weights: α₁=0.55, α₂=0.20, α₃=0.15, α₄=0.10

### Typical Hyperparameters (current lattice training)

- λ (exponential boost strength): 5.0 – 8.0  
- σ_valence: batch std of valences (adaptive)  
- T (temperature): 4.0 – 8.0  
- λ₁ = 0.7–0.9 (soft KD weight)  
- λ₂ = 0.3–0.5 (feature KD weight)  
- λ₃ = 0.2–0.4 (sequence KD weight)  
- λ₄ = 0.4–0.7 (future valence weight)  
- λ₅ = 1e-5 – 1e-4 (weight decay)

## 2. Mercy Gate & Rejection Criteria

During training loop:

if projected_future_valence_after_update < 0.90 × teacher_baseline_valence:
    reject update / revert checkpoint
    log "Mercy gate blocked – low projected valence trajectory"

Projected future valence estimated via:
- Teacher model forward pass on current student state  
- Average over next 10–30 simulated steps  
- Or simpler proxy: current batch average valence × 0.95

## 3. Lattice-Specific Implementation Patterns

### PyTorch training loop excerpt

```python
def compute_valence_weight(valences: torch.Tensor) -> torch.Tensor:
    mu = valences.mean()
    sigma = valences.std() + 1e-8
    return torch.exp(6.0 * (valences - mu) / sigma)

# Inside training step
w = compute_valence_weight(batch_valences)

loss_kd_soft = kl_div_loss(
    F.log_softmax(student_logits / T, dim=1),
    F.softmax(teacher_logits / T, dim=1)
) * (T ** 2)

loss_valence = F.mse_loss(student_future_v, teacher_future_v)

total_loss = (loss_ce + 0.8 * loss_kd_soft + 0.6 * loss_valence) * w.mean()

# Optional: projected future valence check (simplified)
future_v_sim = simulate_future_valence(student_model, batch_inputs)
if future_v_sim.mean() < 0.90 * teacher_future_v_sim.mean():
    print("Mercy gate: projected valence drop detected – skipping update")
    continue
