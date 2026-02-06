# HPA + VPA + Cluster Autoscaler – Combined Mercy Strategy

## Recommended Order of Scaling
1. Vertical Pod Autoscaler (VPA) – first optimize pod resources (CPU/memory)
2. Horizontal Pod Autoscaler (HPA) – scale number of pods once pods are correctly sized
3. Cluster Autoscaler – add/remove nodes when cluster capacity is insufficient

## Mercy Gates & Valence Influence
- Add custom admission webhook (future strike) to prevent VPA/HPA scaling during low-valence periods
- Use node affinity / taints to reserve high-valence nodes for critical inference workloads
- Monitor custom metric: `rathor_valence_spikes_per_minute` → scale proactively on valence surges

## Monitoring & Alerts
- Prometheus + Grafana dashboard for:
  - Pod CPU/Memory usage vs requests/limits
  - Inference latency & throughput
  - Valence spikes & collective trajectory
  - Node pool size & unschedulable pods

## Next Mercy Refinements
- Add Cluster Autoscaler priority expander config
- Integrate Karpenter (AWS-native CA alternative) for faster node provisioning
- Add vertical pod autoscaler status dashboard widget
