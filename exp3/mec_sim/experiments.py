from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .models import SimulationSummary, SystemConfig
from .plotting import plot_bar_comparison, plot_line, plot_multi_line
from .simulation import Simulator
from .utils import ensure_dir, save_json


def save_summaries_to_excel(summaries: List[SimulationSummary], path: str) -> None:
    rows = []
    for summary in summaries:
        row = {
            "policy": summary.name,
            "V": summary.config.get("V"),
            "avg_total_cost": summary.avg_total_cost,
            "avg_operating_cost": summary.avg_operating_cost,
            "avg_local_cost": summary.avg_local_cost,
            "avg_tx_cost": summary.avg_tx_cost,
            "avg_edge_cost": summary.avg_edge_cost,
            "avg_overdue_cost": summary.avg_overdue_cost,
            "total_cost": summary.total_cost,
            "total_operating_cost": summary.total_operating_cost,
            "total_overdue_cost": summary.total_overdue_cost,
            "avg_total_backlog": summary.avg_total_backlog,
            "avg_local_backlog": summary.avg_local_backlog,
            "avg_edge_backlog": summary.avg_edge_backlog,
            "avg_overdue_local_bits": summary.avg_overdue_local_bits,
            "avg_overdue_edge_bits": summary.avg_overdue_edge_bits,
            "avg_overdue_total_bits": summary.avg_overdue_total_bits,
            "total_overdue_local_bits": summary.total_overdue_local_bits,
            "total_overdue_edge_bits": summary.total_overdue_edge_bits,
            "total_overdue_bits": summary.total_overdue_bits,
            "avg_offloaded_bits": summary.avg_offloaded_bits,
            "offload_ratio": summary.offload_ratio,
            "overdue_ratio": summary.overdue_ratio,
            "timeout_ratio": summary.timeout_ratio,
            "avg_sojourn_slots": summary.avg_sojourn_slots,
        }
        rows.append(row)

    pd.DataFrame(rows).to_excel(path, index=False)


class ExperimentManager:
    def __init__(self, base_config: Optional[SystemConfig] = None):
        self.base_config = base_config or SystemConfig()
        ensure_dir(self.base_config.output_dir)

    def _run_policy_group(self, config: SystemConfig, policies: List[str], tag: str) -> List[SimulationSummary]:
        summaries = []
        exp_dir = f"{config.output_dir}/{tag}"
        ensure_dir(exp_dir)

        for name in policies:
            sim = Simulator(config, name)
            summary, _ = sim.run(verbose=True)
            summaries.append(summary)
            save_json(f"{exp_dir}/{name}_summary.json", asdict(summary))
        return summaries

    def experiment_1_algorithm_comparison(self) -> List[SimulationSummary]:
        cfg = SystemConfig(**asdict(self.base_config))
        tag = "exp1_algorithm_comparison"
        policies = ["ADDO", "ADDO-SQ", "DAEE", "LocalOnly", "EdgeOnly", "Random"]
        summaries = self._run_policy_group(cfg, policies, tag)
        exp_dir = f"{cfg.output_dir}/{tag}"
        save_summaries_to_excel(summaries, f"{exp_dir}/experiment_1_summary.xlsx")
        plot_bar_comparison(summaries, "avg_total_cost", "Experiment 1: Average Total Cost", "Average cost per slot", f"{exp_dir}/avg_total_cost.png")
        plot_bar_comparison(summaries, "avg_total_backlog", "Experiment 1: Average Total Backlog", "Average backlog (bits)", f"{exp_dir}/avg_total_backlog.png")
        plot_bar_comparison(summaries, "avg_overdue_total_bits", "Experiment 1: Average Overdue Usage", "Average overdue bits/slot", f"{exp_dir}/avg_overdue_bits.png")
        return summaries

    def experiment_2_v_tradeoff(self, V_list: List[float], policy: str = "ADDO") -> List[SimulationSummary]:
        tag = "exp2_v_tradeoff"
        exp_dir = f"{self.base_config.output_dir}/{tag}"
        ensure_dir(exp_dir)
        summaries = []
        costs = []
        backlogs = []
        overdue = []

        for V in V_list:
            cfg = SystemConfig(**asdict(self.base_config))
            cfg.V = V
            sim = Simulator(cfg, policy)
            summary, _ = sim.run(verbose=True)
            summaries.append(summary)
            costs.append(summary.avg_operating_cost)
            backlogs.append(summary.avg_total_backlog)
            overdue.append(summary.total_overdue_bits)
            save_json(f"{exp_dir}/{policy}_V_{V}_summary.json", asdict(summary))
        return summaries


    def experiment_3_deadline_comparison_at_fixed_v(
        self,
        V: float,
        deadlines: List[int],
        policies: Optional[List[str]] = None,
    ) -> Dict[int, List[SimulationSummary]]:
        tag = "exp3_deadline_comparison_fixed_v"
        exp_dir = f"{self.base_config.output_dir}/{tag}"
        ensure_dir(exp_dir)
        policies = policies or ["ADDO", "ADDO-SQ", "DAEE", "LocalOnly", "EdgeOnly", "Random"]
        results_by_deadline: Dict[int, List[SimulationSummary]] = {}

        for deadline in deadlines:
            cfg = SystemConfig(**asdict(self.base_config))
            cfg.V = V
            cfg.heterogeneous_deadline = True
            cfg.deadline_min = 2
            cfg.deadline_max = deadline

            summaries: List[SimulationSummary] = []
            for policy in policies:
                sim = Simulator(cfg, policy)
                summary, _ = sim.run(verbose=True)
                summaries.append(summary)
                save_json(f"{exp_dir}/{policy}_V_{V}_deadline_{deadline}.json", asdict(summary))

            results_by_deadline[deadline] = summaries
            save_summaries_to_excel(summaries, f"{exp_dir}/deadline_{deadline}_summary.xlsx")
        return results_by_deadline


def print_summary_table(summaries: List[SimulationSummary]) -> None:
    def format_policy_label(summary: SimulationSummary) -> str:
        if summary.name not in {"ADDO", "ADDO-SQ", "DAEE"}:
            return summary.name
        v_value = summary.config.get("V")
        if v_value is None:
            return summary.name
        return f"{summary.name} (V={v_value:.2e})"

    header = (
        f"{'Policy':<28} {'TotalCost':>12} {'OpCost':>12} {'LocalCost':>12} {'TxCost':>12} {'EdgeCost':>12} "
        f"{'TotalRec':>12} {'LocalRec':>12} {'EdgeRec':>12} "
        f"{'OffloadRatio':>14} {'OverdueRatio':>14} {'Backlog':>14} {'AvgDelay':>12}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        policy_label = format_policy_label(summary)
        print(
            f"{policy_label:<28} {summary.avg_total_cost:>12.4f} {summary.avg_operating_cost:>12.5f} "
            f"{summary.avg_local_cost:>12.5f} {summary.avg_tx_cost:>12.5f} {summary.avg_edge_cost:>12.5f} "
            f"{summary.avg_overdue_total_bits:>12.4f} {summary.avg_overdue_local_bits:>12.2f} "
            f"{summary.avg_overdue_edge_bits:>12.2f} {summary.offload_ratio:>14.4f} {summary.overdue_ratio:>14.4f} "
            f"{summary.avg_total_backlog:>14.2f} {summary.avg_sojourn_slots:>12.4f}"
        )


def run_experiment_1() -> None:
    manager = ExperimentManager(SystemConfig())
    summaries = manager.experiment_1_algorithm_comparison()
    print("\n[Experiment 1] Algorithm comparison")
    print_summary_table(summaries)


def run_experiment_2(policy: str = "ADDO", v_values: list[float] = None) -> None:
    manager = ExperimentManager(SystemConfig())
    if v_values is None:
        v_values = [1e8, 2e8, 5e8, 1e9, 2e9, 5e9, 1e10, 2e10, 5e10, 1e11, 1.2e11, 1.5e11, 2e11]
    summaries = manager.experiment_2_v_tradeoff(v_values, policy=policy)
    print(f"\n[Experiment 2] V tradeoff for policy={policy}")
    print_summary_table(summaries)


def run_experiment_3(
    V: float = 1.0e11,
    deadlines: Optional[List[int]] = None,
    policies: Optional[List[str]] = None,
) -> None:
    manager = ExperimentManager(SystemConfig())
    deadlines = deadlines or [2, 3, 4, 5, 6, 8, 10]
    results = manager.experiment_3_deadline_comparison_at_fixed_v(V, deadlines, policies=policies)
    print(f"\n[Experiment 3] Algorithm comparison under different deadlines at fixed V={V}")
    for deadline in deadlines:
        print(f"\nDeadline: {deadline}")
        print_summary_table(results[deadline])


def run_all_experiments() -> None:
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
