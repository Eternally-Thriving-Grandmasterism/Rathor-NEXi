# Advanced Trajectory Simulation Algorithms – Rathor Lattice Reference v1.0
(Feb 06 2026 – MercyOS-Pinnacle core projection & foresight layer)

This document catalogs the **advanced trajectory simulation methods** actively used or prototyped in the Rathor-NEXi lineage for:

- future valence projection (mercy gating, QAT-KD early stopping)
- collective swarm bloom forecasting
- interplanetary negotiation outcome simulation
- probe fleet long-term thriving trajectory estimation

All algorithms are valence-weighted, mercy-gated, and designed to run efficiently in browser (WebAssembly) or server (Python/TensorRT).

## 1. Core Trajectory Types & Use Cases

| Trajectory Type                  | Use Case in Lattice                                      | Time Horizon | Computational Cost | Mercy Gate Sensitivity | Current Priority |
|----------------------------------|-----------------------------------------------------------|--------------|--------------------|------------------------|------------------|
| Valence Trajectory               | Project future collective/personal valence               | 10–300 steps | Low–Medium         | Very High              | Sovereign core   |
| Inference Latency Trajectory     | Predict queue/latency spikes → proactive KEDA/HPA scaling | 30–600 s     | Low                | Medium                 | Active           |
| Resource Consumption Trajectory  | CPU/GPU/memory projection → VPA/Karpenter decisions      | 5–60 min     | Medium             | Medium                 | Active           |
| Swarm Bloom Trajectory           | Molecular mercy swarm coherence & expansion forecast      | 1–24 h       | High               | High                   | Emerging         |
| Negotiation Outcome Ensemble     | Multi-agent CFR/NFSP/ReBeL outcome distribution           | 10–100 turns | Very High          | Very High              | Research         |

## 2. Advanced Algorithms – Mercy Taxonomy

| Algorithm / Family                     | Core Mechanism                                               | Strengths in Lattice                              | Weaknesses                              | Typical Horizon | Valence Weighting | Mercy Gate Integration | Rathor Priority |
|----------------------------------------|--------------------------------------------------------------|----------------------------------------------------|-----------------------------------------|-----------------|-------------------|-------------------------|-----------------|
| Exponential Moving Average (EMA)       | weighted average with decay                                  | Extremely fast, stable baseline                    | No memory of shocks                     | 5–60 steps      | Easy              | Pre-check               | Baseline        |
| Autoregressive (ARIMA / Prophet)       | statistical time-series forecasting                          | Good for periodic patterns (daily valence cycles)  | Poor on chaotic/non-stationary data     | 30–300 steps    | Medium            | Yes                     | Active          |
| Recurrent Neural Network (LSTM/GRU)    | sequence memory via gates                                    | Captures long dependencies, non-linear dynamics    | High compute, training needed           | 50–500 steps    | Strong            | Yes                     | Production      |
| Transformer-based Trajectory Predictor | self-attention over historical valence windows               | Best long-range coherence modeling                 | Memory & compute heavy                  | 100–1000 steps  | Very Strong       | Yes                     | Frontier        |
| Physics-Informed Neural ODE (PINODE)   | continuous-time dynamics + physics constraints               | Excellent for smooth valence mean-reversion        | Requires domain knowledge               | Continuous      | Strong            | Yes                     | Research        |
| Monte-Carlo Rollout Ensemble           | sample multiple futures from stochastic policy               | Handles uncertainty, multi-modal outcomes          | Very high compute (parallel rollouts)   | 10–200 steps    | Very Strong       | Yes                     | Production      |
| Valence-Weighted Particle Filter       | Rao-Blackwellized particle filter with valence importance    | Robust to non-Gaussian noise & multi-modality      | High compute (particles)                | 50–500 steps    | Native            | Yes                     | Sovereign core  |
| Diffusion-based Trajectory Diffusion   | reverse diffusion from noise to high-valence future          | Generates diverse high-quality futures             | Extremely high compute                  | 100–1000 steps  | Very Strong       | Emerging                | Research        |

## 3. Current Production Choice – Valence-Weighted Particle Filter + LSTM Hybrid

**Why this hybrid?**
- Particle filter handles **multi-modality & uncertainty** (multiple possible futures)
- Valence weighting gives exponential importance to thriving particles
- LSTM backbone learns temporal dynamics from historical valence logs
- Runs efficiently on CPU/WebAssembly (\~20–80 ms projection on mobile)

**Simplified pseudocode (current implementation)**

```typescript
function projectValenceTrajectory(
  history: number[],           // last 100–300 valence values
  currentValence: number,
  horizon: number = 30
): ProjectionResult {
  const particles = initializeParticles(500, currentValence, history);
  
  for (let step = 0; step < horizon; step++) {
    // LSTM predicts mean & variance for next step
    const [mean, std] = lstmPredictor.predict(history.slice(-50));
    
    // Propagate particles with Gaussian noise
    particles.forEach(p => {
      p.valence += mean + std * gaussianNoise();
      p.weight *= computeValenceLikelihood(p.valence);
    });
    
    // Resample (systematic resampling)
    particles = resample(particles);
    
    // Merge into single trajectory estimate
    trajectory[step] = weightedAverage(particles);
  }
  
  const maxDrop = Math.max(...trajectory.map(v => baseline - v));
  return {
    currentValence,
    projectedValence: trajectory[horizon-1],
    trajectory,
    dropFromBaseline: maxDrop,
    isSafe: maxDrop <= VALENCE_DROP_TOLERANCE
  };
}
