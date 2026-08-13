from __future__ import annotations

from typing import Dict

import numpy as np

from .models import StepMetrics, SystemConfig, UserParams


class MECEnvironment:
    def __init__(self, config: SystemConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)
        self.user_params = self._build_user_params()
        self.K = config.num_users
        self.Dmax = int(np.max(self.user_params.deadline))
        self.QL = np.zeros((self.K, self.Dmax), dtype=float)
        self.QE = np.zeros((self.K, self.Dmax), dtype=float)
        self.ZL = np.zeros(self.K, dtype=float)
        self.ZE = np.zeros(self.K, dtype=float)
        self.L_stamp = np.full((self.K, self.Dmax), np.nan)
        self.E_stamp = np.full((self.K, self.Dmax), np.nan)
        self.t = 0

    def _build_user_params(self) -> UserParams:
        cfg = self.cfg
        rng = self.rng
        K = cfg.num_users

        cpb = rng.integers(cfg.cycles_per_bit_min, cfg.cycles_per_bit_max + 1, size=K)
        local_cpu_max = np.full(K, cfg.local_cpu_max, dtype=float)
        tx_power_max = np.full(K, cfg.tx_power_max, dtype=float)
        bandwidth = np.full(K, cfg.bandwidth, dtype=float)

        if cfg.heterogeneous_deadline:
            deadline = rng.integers(cfg.deadline_min, cfg.deadline_max + 1, size=K)
        else:
            deadline = np.full(K, cfg.deadline_max, dtype=int)

        if cfg.heterogeneous_channel:
            distances = rng.uniform(0.8 * cfg.user_distance, 1.2 * cfg.user_distance, size=K)
        else:
            distances = np.full(K, cfg.user_distance, dtype=float)

        channel_mean = cfg.reference_channel_gain * (cfg.reference_distance / distances) ** cfg.path_loss_exponent

        return UserParams(
            cycles_per_bit=cpb,
            local_cpu_max=local_cpu_max,
            tx_power_max=tx_power_max,
            bandwidth=bandwidth,
            deadline=deadline,
            channel_mean=channel_mean,
            phi_local=np.full(K, cfg.phi_local, dtype=float),
            phi_edge=np.full(K, cfg.phi_edge, dtype=float),
        )

    def _sample_arrivals(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.arrival_mode == "uniform":
            arr = self.rng.integers(0, cfg.arrival_max_bits + 1, size=self.K)
        elif cfg.arrival_mode == "bursty":
            burst_flag = self.rng.random(self.K) < cfg.burst_probability
            arr = np.where(
                burst_flag,
                self.rng.integers(cfg.burst_high_min_bits, cfg.burst_high_max_bits + 1, size=self.K),
                self.rng.integers(0, cfg.burst_low_max_bits + 1, size=self.K),
            )
        else:
            raise ValueError(f"Unsupported arrival_mode={cfg.arrival_mode}")
        return arr * cfg.arrival_load_scale

    def _sample_channels(self) -> np.ndarray:
        # Frequency-flat block fading: H_i(t) = beta_0 (d_0 / d_i)^alpha xi_i(t),
        # where xi_i(t) is exponential with unit mean under Rayleigh fading.
        return self.rng.exponential(scale=self.user_params.channel_mean, size=self.K)

    def observe(self) -> Dict[str, np.ndarray]:
        return {
            "QL": self.QL.copy(),
            "QE": self.QE.copy(),
            "ZL": self.ZL.copy(),
            "ZE": self.ZE.copy(),
            "QL_total": self.QL.sum(axis=1),
            "QE_total": self.QE.sum(axis=1),
            "arrivals": self._sample_arrivals(),
            "channel": self._sample_channels(),
        }

    @staticmethod
    def _weighted_merge_stamp(old_bits: float, old_stamp: float, add_bits: float, add_stamp: float) -> float:
        if add_bits <= 0:
            return old_stamp
        if old_bits <= 0 or np.isnan(old_stamp):
            return add_stamp
        return (old_bits * old_stamp + add_bits * add_stamp) / (old_bits + add_bits)

    def step(self, action: Dict[str, np.ndarray], obs: Dict[str, np.ndarray]) -> StepMetrics:
        cfg = self.cfg
        up = self.user_params
        fL = action["fL"]
        p = action["p"]
        fE = action["fE"]

        muL = cfg.slot_length * fL / up.cycles_per_bit
        muT = cfg.slot_length * up.bandwidth * np.log2(
            1.0 + p * obs["channel"] / (cfg.noise_density * up.bandwidth)
        )
        muE = cfg.slot_length * fE / up.cycles_per_bit

        metrics = StepMetrics(arrived_bits=float(np.sum(obs["arrivals"])))
        new_QL = np.zeros_like(self.QL)
        new_QE = np.zeros_like(self.QE)
        new_ZL = self.ZL.copy()
        new_ZE = self.ZE.copy()
        new_L_stamp = np.full_like(self.L_stamp, np.nan)
        new_E_stamp = np.full_like(self.E_stamp, np.nan)

        for i in range(self.K):
            D = int(up.deadline[i])
            ql = self.QL[i, :D].copy()
            qe = self.QE[i, :D].copy()
            l_stamp = self.L_stamp[i, :D].copy()
            e_stamp = self.E_stamp[i, :D].copy()
            arr = obs["arrivals"][i]

            x = np.zeros(D)
            remaining = muL[i]
            for a in reversed(range(D)):
                served = min(ql[a], remaining)
                x[a] = served
                remaining -= served

            ql_after_local = ql - x
            u = np.zeros(D)
            remaining = muT[i]
            for a in reversed(range(D - 1)):
                served = min(ql_after_local[a], remaining)
                u[a] = served
                remaining -= served

            y = np.zeros(D)
            remaining = muE[i]
            for a in reversed(range(D)):
                served = min(qe[a], remaining)
                y[a] = served
                remaining -= served

            rL = max(0.0, ql[D - 1] - x[D - 1])
            rE = max(0.0, qe[D - 1] - y[D - 1])
            violation_total = rL + rE
            metrics.completed_bits += float(np.sum(x) + np.sum(y) + rL + rE)

            for a in range(D):
                if x[a] > 0 and not np.isnan(l_stamp[a]):
                    metrics.sojourn_sum += x[a] * (self.t - l_stamp[a] + 1)
                    metrics.sojourn_jobs += x[a]
                if y[a] > 0 and not np.isnan(e_stamp[a]):
                    metrics.sojourn_sum += y[a] * (self.t - e_stamp[a] + 1)
                    metrics.sojourn_jobs += y[a]
            if rL > 0 and not np.isnan(l_stamp[D - 1]):
                metrics.sojourn_sum += rL * (self.t - l_stamp[D - 1] + 1)
                metrics.sojourn_jobs += rL
            if rE > 0 and not np.isnan(e_stamp[D - 1]):
                metrics.sojourn_sum += rE * (self.t - e_stamp[D - 1] + 1)
                metrics.sojourn_jobs += rE

            remain_local = ql - x - u
            for a in range(D - 1):
                new_QL[i, a + 1] = remain_local[a]
                if remain_local[a] > 0 and not np.isnan(l_stamp[a]):
                    new_L_stamp[i, a + 1] = l_stamp[a]

            remain_edge = qe - y
            for a in range(D - 1):
                new_QE[i, a + 1] += remain_edge[a]
                if remain_edge[a] > 0 and not np.isnan(e_stamp[a]):
                    new_E_stamp[i, a + 1] = e_stamp[a]
                if u[a] > 0:
                    existing_bits = new_QE[i, a + 1]
                    existing_stamp = new_E_stamp[i, a + 1]
                    origin_stamp = l_stamp[a] if not np.isnan(l_stamp[a]) else self.t
                    new_stamp = self._weighted_merge_stamp(existing_bits, existing_stamp, u[a], origin_stamp)
                    new_QE[i, a + 1] += u[a]
                    new_E_stamp[i, a + 1] = new_stamp

            # New arrivals occur at the end of slot t and become available
            # in the local age-0 queue starting from slot t + 1.
            if arr > 0:
                new_QL[i, 0] += arr
                new_L_stamp[i, 0] = self._weighted_merge_stamp(new_QL[i, 0] - arr, new_L_stamp[i, 0], arr, self.t + 1)

            offloaded_bits = float(np.sum(u))

            # The paper defines regular operating cost directly from the
            # allocated control actions, not from realized utilization.
            metrics.local_cost += cfg.slot_length * cfg.kappa_local * (fL[i] ** 3)
            metrics.tx_cost += cfg.slot_length * p[i]
            metrics.edge_cost += cfg.slot_length * cfg.kappa_edge * (fE[i] ** 3)
            metrics.overdue_cost += up.phi_local[i] * rL + up.phi_edge[i] * rE
            metrics.overdue_local_bits += rL
            metrics.overdue_edge_bits += rE
            metrics.offloaded_bits += offloaded_bits
            metrics.violation_bits += violation_total

            new_ZL[i] = max(0.0, self.ZL[i] + cfg.persist_local * (1.0 if self.QL.sum(axis=1)[i] > 0 else 0.0) - muL[i] - muT[i])
            new_ZE[i] = max(0.0, self.ZE[i] + cfg.persist_edge * (1.0 if self.QE.sum(axis=1)[i] > 0 else 0.0) - muE[i])

        metrics.total_cost = metrics.local_cost + metrics.tx_cost + metrics.edge_cost + metrics.overdue_cost
        self.QL = new_QL
        self.QE = new_QE
        self.ZL = new_ZL
        self.ZE = new_ZE
        self.L_stamp = new_L_stamp
        self.E_stamp = new_E_stamp
        self.t += 1
        metrics.total_local_backlog = float(np.sum(self.QL))
        metrics.total_edge_backlog = float(np.sum(self.QE))
        metrics.total_backlog = metrics.total_local_backlog + metrics.total_edge_backlog
        metrics.virtual_local_backlog = float(np.sum(self.ZL))
        metrics.virtual_edge_backlog = float(np.sum(self.ZE))
        return metrics
