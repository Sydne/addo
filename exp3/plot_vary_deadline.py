import json
from pathlib import Path

import mec_sim.plotting as plotting


EXP_DIR = Path("results/exp3_deadline_comparison_fixed_v")
V_VALUE = 1e11
DEADLINES = [2, 3, 4, 5, 6, 8, 10]
POLICIES = [
    "ADDO",
    "ADDO-SQ",
    "DAEE",
    "LocalOnly",
    "EdgeOnly",
    "Random",
]


def _format_v_token(v_value: float) -> str:
    return str(float(v_value))


def _format_v_filename_token(v_value: float) -> str:
    mantissa, exponent = f"{float(v_value):.15E}".split("E")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}E{int(exponent)}"


def load_experiment_3_summaries(
    v_value: float,
    deadlines: list[int],
    policies: list[str],
) -> tuple[list[int], dict[str, tuple[list[int], list[float]]], dict[str, tuple[list[int], list[float]]], dict[str, tuple[list[int], list[float]]]]:
    v_token = _format_v_token(v_value)

    violation_series: dict[str, tuple[list[int], list[float]]] = {}
    energy_series: dict[str, tuple[list[int], list[float]]] = {}
    delay_series: dict[str, tuple[list[int], list[float]]] = {}
    if not deadlines:
        raise ValueError("DEADLINES cannot be empty.")
    if not policies:
        raise ValueError("POLICIES cannot be empty.")

    for deadline in deadlines:
        for policy in policies:
            summary_path = EXP_DIR / f"{policy}_V_{v_token}_deadline_{deadline}.json"
            if not summary_path.exists():
                raise FileNotFoundError(
                    f"Missing experiment 3 summary file for policy={policy}, V={v_value}, deadline={deadline}: {summary_path}"
                )

            with open(summary_path, "r", encoding="utf-8") as file:
                summary = json.load(file)

            overdue_ratio = summary.get("overdue_ratio", summary.get("timeout_ratio"))
            if overdue_ratio is None:
                raise KeyError(f"Neither 'overdue_ratio' nor 'timeout_ratio' found in {summary_path}")

            avg_operating_cost = summary.get("avg_operating_cost")
            avg_sojourn_slots = summary.get("avg_sojourn_slots")
            if avg_operating_cost is None or avg_sojourn_slots is None:
                raise KeyError(f"Missing required metrics in {summary_path}")

            violation_series.setdefault(policy, ([], []))
            violation_series[policy][0].append(deadline)
            violation_series[policy][1].append(overdue_ratio)

            energy_series.setdefault(policy, ([], []))
            energy_series[policy][0].append(deadline)
            energy_series[policy][1].append(avg_operating_cost)

            delay_series.setdefault(policy, ([], []))
            delay_series[policy][0].append(deadline)
            delay_series[policy][1].append(avg_sojourn_slots)

    for series_group in (violation_series, energy_series, delay_series):
        for policy, (xs, ys) in series_group.items():
            ordered_pairs = sorted(zip(xs, ys), key=lambda item: item[0])
            series_group[policy] = ([deadline for deadline, _ in ordered_pairs], [value for _, value in ordered_pairs])

    return sorted(deadlines), violation_series, energy_series, delay_series


def main() -> None:
    deadlines, violation_series, energy_series, delay_series = load_experiment_3_summaries(
        V_VALUE,
        DEADLINES,
        POLICIES,
    )
    v_filename_token = _format_v_filename_token(V_VALUE)

    plotting.plot_multi_line(
        violation_series,
        r"$D_{\max}$ (slots)",
        "Deadline violation ratio",
        str(EXP_DIR / f"deadline_violation_ratio_vs_deadline_V_{v_filename_token}.png"),
        legend_columns=2,
        legend_location="center right",
        legend_bbox_to_anchor=(0.99, 0.65),
    )
    plotting.plot_multi_line(
        energy_series,
        r"$D_{\max}$ (slots)",
        "Average energy cost (J/slot)",
        str(EXP_DIR / f"avg_energy_cost_vs_deadline_V_{v_filename_token}.png"),
        legend_columns=2,
        legend_location="upper right",
        legend_bbox_to_anchor=(0.99, 0.93),
    )
    plotting.plot_multi_line(
        delay_series,
        r"$D_{\max}$ (slots)",
        "Average delay (slots)",
        str(EXP_DIR / f"avg_delay_vs_deadline_V_{v_filename_token}.png"),
        legend_columns=2,
    )
    print(f"Plotted Experiment 3 metrics for V={V_VALUE} across deadlines: {deadlines}")


if __name__ == "__main__":
    main()
