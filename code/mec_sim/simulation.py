from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import numpy as np

from .environment import MECEnvironment
from .models import SimulationSummary, StepMetrics, SystemConfig
from .policies import POLICY_MAP


class MetricsCollector:
    def __init__(self, warmup_slots: int):
        self.warmup_slots = warmup_slots
        self.records: List[StepMetrics] = []
        self.all_records: List[StepMetrics] = []

    def add(self, step_idx: int, metrics: StepMetrics) -> None:
        self.all_records.append(metrics)
        if step_idx >= self.warmup_slots:
            self.records.append(metrics)

    def summarize(self, name: str, config: SystemConfig) -> SimulationSummary:
        if not self.records:
            raise RuntimeError("No metrics collected.")

        def avg(attr: str) -> float:
            return float(np.mean([getattr(record, attr) for record in self.records]))

        def total(attr: str) -> float:
            return float(np.sum([getattr(record, attr) for record in self.records]))

        total_arrived = total("arrived_bits")
        total_offloaded = total("offloaded_bits")
        sojourn_sum = total("sojourn_sum")
        sojourn_jobs = total("sojourn_jobs")
        total_local_cost = total("local_cost")
        total_tx_cost = total("tx_cost")
        total_edge_cost = total("edge_cost")
        total_overdue_cost = total("overdue_cost")
        total_operating_cost = total_local_cost + total_tx_cost + total_edge_cost
        total_cost = total_operating_cost + total_overdue_cost
        total_overdue_local_bits = total("overdue_local_bits")
        total_overdue_edge_bits = total("overdue_edge_bits")
        overdue_ratio = (total_overdue_local_bits + total_overdue_edge_bits) / max(total_arrived, 1e-12)

        return SimulationSummary(
            name=name,
            config=asdict(config),
            avg_total_cost=avg("total_cost"),
            avg_operating_cost=avg("local_cost") + avg("tx_cost") + avg("edge_cost"),
            avg_local_cost=avg("local_cost"),
            avg_tx_cost=avg("tx_cost"),
            avg_edge_cost=avg("edge_cost"),
            avg_overdue_cost=avg("overdue_cost"),
            total_cost=total_cost,
            total_operating_cost=total_operating_cost,
            total_overdue_cost=total_overdue_cost,
            avg_total_backlog=avg("total_backlog"),
            avg_local_backlog=avg("total_local_backlog"),
            avg_edge_backlog=avg("total_edge_backlog"),
            avg_overdue_local_bits=avg("overdue_local_bits"),
            avg_overdue_edge_bits=avg("overdue_edge_bits"),
            avg_overdue_total_bits=avg("overdue_local_bits") + avg("overdue_edge_bits"),
            total_overdue_local_bits=total_overdue_local_bits,
            total_overdue_edge_bits=total_overdue_edge_bits,
            total_overdue_bits=total_overdue_local_bits + total_overdue_edge_bits,
            avg_offloaded_bits=avg("offloaded_bits"),
            offload_ratio=total_offloaded / max(total_arrived, 1e-12),
            overdue_ratio=overdue_ratio,
            timeout_ratio=overdue_ratio,
            avg_sojourn_slots=sojourn_sum / max(sojourn_jobs, 1e-12),
        )


class Simulator:
    def __init__(self, config: SystemConfig, policy_name: str):
        if policy_name not in POLICY_MAP:
            raise ValueError(f"Unknown policy: {policy_name}")
        self.config = config
        self.env = MECEnvironment(config)
        self.policy = POLICY_MAP[policy_name](config, self.env.user_params)
        self.policy_name = policy_name

    def run(self, verbose: bool = False) -> Tuple[SimulationSummary, MetricsCollector]:
        collector = MetricsCollector(self.config.warmup_slots)
        for t in range(self.config.num_slots):
            obs = self.env.observe()
            action = self.policy.act(obs)
            metrics = self.env.step(action, obs)
            collector.add(t, metrics)
            if verbose and (t + 1) % 5000 == 0:
                print(f"[{self.policy_name}] slot {t + 1}/{self.config.num_slots}")
        return collector.summarize(self.policy_name, self.config), collector
