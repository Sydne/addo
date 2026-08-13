import json
from pathlib import Path

import mec_sim.plotting as plotting


exp_dir = Path("results/exp2_v_tradeoff")
policies = ["ADDO", "ADDO-SQ", "DAEE"]
metrics = {
    "avg_operating_cost": "Average energy cost (J)",
    "avg_sojourn_slots": "Average delay (slots)",
    "timeout_ratio": "Deadline violation ratio",
}
series = {policy: {metric: [] for metric in metrics} for policy in policies}


def available_v_values(policy: str) -> dict[float, Path]:
    files = {}
    for path in exp_dir.glob(f"{policy}_V_*_summary.json"):
        v_str = path.name.removeprefix(f"{policy}_V_").removesuffix("_summary.json")
        files[float(v_str)] = path
    return files


policy_files = {policy: available_v_values(policy) for policy in policies}
common_v_list = [1e8, 2e8, 5e8, 1e9, 2e9, 5e9, 1e10, 2e10, 5e10, 1e11, 1.2e11, 1.5e11, 2e11]

for policy in policies:
    missing = [v for v in common_v_list if v not in policy_files[policy]]
    if missing:
        raise FileNotFoundError(
            f"Missing Experiment 2 summaries for policy={policy}, V values={missing} in {exp_dir}"
        )
    for v in common_v_list:
        with open(policy_files[policy][v], "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        for metric in metrics:
            series[policy][metric].append(summary_data[metric])

plotting.plot_multi_line(
    {policy: (common_v_list, series[policy]["avg_operating_cost"]) for policy in policies},
    "V",
    metrics["avg_operating_cost"],
    str(exp_dir / "ADDO_vs_DAEE_avg_energy_cost_vs_V.png"),
)
plotting.plot_multi_line(
    {policy: (common_v_list, series[policy]["avg_sojourn_slots"]) for policy in policies},
    "V",
    metrics["avg_sojourn_slots"],
    str(exp_dir / "ADDO_vs_DAEE_avg_delay_slots_vs_V.png"),
)
plotting.plot_multi_line(
    {policy: (common_v_list, series[policy]["timeout_ratio"]) for policy in policies},
    "V",
    metrics["timeout_ratio"],
    str(exp_dir / "ADDO_vs_DAEE_deadline_violation_ratio_vs_V.png"),
)
