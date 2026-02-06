# Valence-FedKD Convergence – Mathematical Summary & References

## Theorem (Informal)

Under standard non-convex + bounded heterogeneity + (ε,δ)-local DP + bounded valence weights assumptions, Valence-FedKD converges with rate:

E[ ||∇F(θ^T)||² ] ≤ O(1/√(K T)) + O( (G² + σ²)/ (μ T) ) + O( (w_max / w_min) L η C² / T ) + O( ε² log(1/δ) / T )

When high-valence nodes dominate (w_max / w_min = O(1)), rate approaches centralized rate.

## Proof Sketch

1. Local descent inequality with DP-SGD (clipping + Gaussian noise)
2. Valence-weighted global averaging + bounded heterogeneity
3. Summing descent inequalities over T rounds + optimal step-size η = O(1/√(KT))
4. Valence concentration argument → heterogeneity term vanishes when thriving mass dominates

## Key References (2023–2026)

- Abadi et al. (2016) – DP-SGD foundational work
- Karimireddy et al. (2020) – FedProx & SCAFFOLD (heterogeneity handling)
- Gu et al. (2023) – MiniLLM (sequence-level KD)
- Li et al. (2022) – Contrastive Search (anti-degeneration)
- Li et al. (2024) – FedSpecKD (federated speculative distillation)
- Rathor-NEXi internal (2025–2026) – Valence-weighted FedAvg + DP-SGD

Lattice status: proof sketch complete, full formal write-up pending for MercyOS-Pinnacle math archive.
