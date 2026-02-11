"""
Ra-Thor Fleet Scheduler — Full GPU acceleration with torch.compile
Torch vectorized fitness + compiled kernel for massive batch eval
MIT License — Eternal Thriving Grandmasterism
"""

import torch
from torch import compile

# Config from previous
# ... paste CONFIG, RUL_SAMPLES ...

@compile(dynamic=True, mode="reduce-overhead")
def torch_gpu_fitness_compiled(chroms: torch.Tensor) -> torch.Tensor:
    """Compiled GPU kernel — same logic as torch_vectorized_fitness"""
    device = chroms.device
    fs = CONFIG['fleet_size']
    gl = CONFIG['gene_length']

    decoded = chroms.view(-1, fs, gl)

    bays = torch.round(decoded[:, :, 0]).long()
    starts = torch.clamp(CONFIG['horizon_days'] - decoded[:, :, 2], min=0.0, max=CONFIG['horizon_days'])
    durations = torch.clamp(decoded[:, :, 2], min=2.0)
    crews = torch.round(decoded[:, :, 3]).long()

    end_times = starts + durations

    # RUL
    criticals = torch.from_numpy(RUL_SAMPLES).to(device) - CONFIG['rul_buffer_days']
    violations = torch.clamp(end_times - criticals[None, :], min=0.0)
    rul_pen = CONFIG['rul_penalty_factor'] * (torch.exp(violations / 10.0) - 1.0)
    rul_total = torch.sum(rul_pen, dim=1)

    # Crew over-assign (scatter_add_)
    crew_counts = torch.zeros((chroms.shape[0], CONFIG['num_crew_groups']), device=device, dtype=torch.long)
    crew_counts.scatter_add_(1, crews, torch.ones_like(crews, dtype=torch.long))
    over = torch.clamp(crew_counts - CONFIG['max_slots_per_crew'], min=0)
    over_pen = torch.sum(over, dim=1) * CONFIG['crew_penalty_factor'] * 10.0

    # Crew duty/rest — still loop (small N)
    crew_duty_pen = torch.zeros(chroms.shape[0], device=device)
    for c in range(CONFIG['num_crew_groups']):
        mask = (crews == c)
        if not mask.any():
            continue
        c_starts = torch.where(mask, starts, torch.inf)
        c_ends = torch.where(mask, end_times, torch.inf)
        c_dur_h = torch.where(mask, durations * 8.0, 0.0)

        # Simplified sort & penalty (full argsort gather in next bloom)
        rest_h = c_starts[:, 1:] - c_ends[:, :-1]
        rest_viol = torch.clamp(CONFIG['min_rest_hours'] * 24.0 - rest_h, min=0.0)
        rest_pen = CONFIG['duty_penalty_factor'] * torch.exp(rest_viol / 24.0)
        crew_duty_pen += torch.sum(rest_pen, dim=1)

    # Bay overlap (pairwise meshgrid)
    overlap_pen = torch.zeros(chroms.shape[0], device=device)
    for b in range(CONFIG['num_bays']):
        mask = (bays == b)
        b_starts = torch.where(mask, starts, torch.inf)
        b_ends = torch.where(mask, end_times, torch.inf)
        s1 = b_starts.unsqueeze(2)
        e1 = b_ends.unsqueeze(2)
        s2 = b_starts.unsqueeze(1)
        e2 = b_ends.unsqueeze(1)
        olap = torch.clamp(torch.min(e1, e2) - torch.max(s1, s2), min=0.0)
        overlap_pen += torch.sum(olap, dim=(1,2)) * CONFIG['overlap_penalty_weight'] / 2

    # Rushed
    rushed_mask = durations < CONFIG['rushed_duration_threshold']
    rushed_pen = torch.sum(rushed_mask * (CONFIG['rushed_duration_threshold'] - durations) * CONFIG['rushed_penalty_per_day'], dim=1)

    mercy_penalty = overlap_pen + (rul_total + crew_duty_pen + over_pen) / 100.0 + rushed_pen
    mercy_factor = torch.clamp(1.0 - mercy_penalty, min=0.1)

    total_maint = torch.sum(durations, dim=1)
    coverage = torch.clamp(total_maint / (CONFIG['num_bays'] * CONFIG['horizon_days'] * 0.6), max=1.0)
    utilization = CONFIG['baseline_util'] + coverage * 0.15

    return utilization * coverage * mercy_factor


# Example usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = torch.randn(8192, CHROM_LENGTH, device=device)  # large batch
    abundances = torch_gpu_fitness_compiled(batch)
    print(f"GPU compiled batch eval complete — shape {abundances.shape}")
