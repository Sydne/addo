from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SystemConfig:
    num_users: int = 20
    slot_length: float = 0.01  # seconds
    num_slots: int = 10000
    warmup_slots: int = 0
    random_seed: int = 42
    deadline_min: int = 2
    deadline_max: int = 10
    heterogeneous_deadline: bool = True
    arrival_mode: str = "uniform"
    arrival_max_bits: float = 3.0e4  # bits
    burst_low_max_bits: float = 0.6e6
    burst_high_min_bits: float = 1.0e6
    burst_high_max_bits: float = 1.8e6
    burst_probability: float = 0.25
    arrival_load_scale: float = 1.0
    cycles_per_bit_min: float = 500
    cycles_per_bit_max: float = 1000
    local_cpu_max: float = 1.0e9
    edge_cpu_max_total: float = 10.0e9
    tx_power_max: float = 0.5  # Watts (100 mW)
    bandwidth: float = 1.0e6  # Hz (1 MHz)
    noise_density: float = 3.981071705534986e-21  # Watts/Hz (-174 dBm/Hz)
    channel_mode: str = "rayleigh"
    reference_distance: float = 1.0  # meters
    reference_channel_gain: float = 1.0e-4  # beta_0, linear scale (-40 dB)
    path_loss_exponent: float = 4.0
    user_distance: float = 150.0  # meters
    heterogeneous_channel: bool = False
    kappa_local: float = 1.0e-28
    kappa_edge: float = 1.0e-28
    phi_local: float = 1.0e-1
    phi_edge: float = 1.0e-1
    persist_local: float = 2.0e4
    persist_edge: float = 2.0e4
    V: float = 1.0e11
    bisection_tol: float = 1.0e-5
    bisection_max_iter: int = 80
    output_dir: str = "results"

    def __post_init__(self) -> None:
        if self.num_users <= 0:
            raise ValueError("num_users must be positive")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if not 0 <= self.warmup_slots < self.num_slots:
            raise ValueError("warmup_slots must satisfy 0 <= warmup_slots < num_slots")
        if self.deadline_min <= 0 or self.deadline_max < self.deadline_min:
            raise ValueError("deadlines must satisfy 0 < deadline_min <= deadline_max")
        if self.slot_length <= 0:
            raise ValueError("slot_length must be positive")
        if self.arrival_mode not in {"uniform", "bursty"}:
            raise ValueError("arrival_mode must be 'uniform' or 'bursty'")
        if self.arrival_load_scale < 0:
            raise ValueError("arrival_load_scale must be non-negative")
        if self.cycles_per_bit_min <= 0 or self.cycles_per_bit_max < self.cycles_per_bit_min:
            raise ValueError("cycles-per-bit bounds must be positive and ordered")
        if min(self.local_cpu_max, self.edge_cpu_max_total, self.tx_power_max, self.bandwidth) < 0:
            raise ValueError("resource limits must be non-negative")
        if self.bandwidth == 0 or self.noise_density <= 0:
            raise ValueError("bandwidth and noise_density must be positive")
        if self.user_distance <= 0 or self.reference_distance <= 0 or self.reference_channel_gain <= 0:
            raise ValueError("channel distances and reference gain must be positive")
        if not 0 <= self.burst_probability <= 1:
            raise ValueError("burst_probability must be between 0 and 1")
        if self.bisection_tol <= 0 or self.bisection_max_iter <= 0:
            raise ValueError("bisection settings must be positive")


@dataclass
class UserParams:
    cycles_per_bit: np.ndarray
    local_cpu_max: np.ndarray
    tx_power_max: np.ndarray
    bandwidth: np.ndarray
    deadline: np.ndarray
    channel_mean: np.ndarray
    phi_local: np.ndarray
    phi_edge: np.ndarray


@dataclass
class StepMetrics:
    total_cost: float = 0.0
    local_cost: float = 0.0
    tx_cost: float = 0.0
    edge_cost: float = 0.0
    overdue_cost: float = 0.0
    overdue_local_bits: float = 0.0
    overdue_edge_bits: float = 0.0
    total_local_backlog: float = 0.0
    total_edge_backlog: float = 0.0
    total_backlog: float = 0.0
    offloaded_bits: float = 0.0
    arrived_bits: float = 0.0
    completed_bits: float = 0.0
    violation_bits: float = 0.0
    virtual_local_backlog: float = 0.0
    virtual_edge_backlog: float = 0.0
    virtual_violation_backlog: float = 0.0
    sojourn_sum: float = 0.0
    sojourn_jobs: float = 0.0


@dataclass
class SimulationSummary:
    name: str
    config: Dict
    avg_total_cost: float
    avg_operating_cost: float
    avg_local_cost: float
    avg_tx_cost: float
    avg_edge_cost: float
    avg_overdue_cost: float
    total_cost: float
    total_operating_cost: float
    total_overdue_cost: float
    avg_total_backlog: float
    avg_local_backlog: float
    avg_edge_backlog: float
    avg_overdue_local_bits: float
    avg_overdue_edge_bits: float
    avg_overdue_total_bits: float
    total_overdue_local_bits: float
    total_overdue_edge_bits: float
    total_overdue_bits: float
    avg_offloaded_bits: float
    offload_ratio: float
    overdue_ratio: float
    timeout_ratio: float
    avg_sojourn_slots: float
