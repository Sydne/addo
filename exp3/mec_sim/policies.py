from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .models import SystemConfig, UserParams
from .utils import clipped_sqrt


class BasePolicy:
    def __init__(self, config: SystemConfig, user_params: UserParams):
        self.cfg = config
        self.up = user_params
        self.rng = np.random.default_rng(config.random_seed)

    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class ADDOPolicy(BasePolicy):
    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        up = self.up
        K = len(up.deadline)
        QL_total = obs["QL_total"]
        QE_total = obs["QE_total"]
        channel = obs["channel"]
        QL = obs["QL"]
        QE = obs["QE"]

        fL = np.zeros(K)
        p = np.zeros(K)

        for i in range(K):
            D = int(up.deadline[i])
            q_last = QL[i, D - 1]
            ci = up.cycles_per_bit[i]
            threshold = ci * q_last / cfg.slot_length

            cand1 = clipped_sqrt(up.phi_local[i] / (3.0 * cfg.kappa_local * ci))
            cand1 = np.clip(cand1, 0.0, min(up.local_cpu_max[i], threshold))
            cand2 = clipped_sqrt(QL_total[i] / (3.0 * cfg.V * cfg.kappa_local * ci))
            cand2 = np.clip(cand2, threshold, up.local_cpu_max[i])

            def local_objective(freq: float) -> float:
                mu = cfg.slot_length * freq / ci
                rec = max(0.0, q_last - mu)
                return (
                    cfg.V * cfg.slot_length * cfg.kappa_local * freq**3
                    - cfg.slot_length * QL_total[i] * freq / ci
                    + (cfg.V * up.phi_local[i] - QL_total[i]) * rec
                )

            candidates = [cand1, cand2]
            values = [local_objective(candidate) for candidate in candidates]
            fL[i] = candidates[int(np.argmin(values))]

            bandwidth = up.bandwidth[i]
            term = (
                (QL_total[i] - QE_total[i]) * bandwidth / (cfg.V * math.log(2.0))
                - cfg.noise_density * bandwidth / max(channel[i], 1e-12)
            )
            p[i] = np.clip(term, 0.0, up.tx_power_max[i])

        return {"fL": fL, "p": p, "fE": self._solve_edge_cpu(obs)}

    def _solve_edge_cpu(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        cfg = self.cfg
        up = self.up
        QE_total = obs["QE_total"]
        QE = obs["QE"]
        K = len(up.deadline)

        def best_f_for_user(i: int, lam: float) -> float:
            D = int(up.deadline[i])
            q_last = QE[i, D - 1]
            ci = up.cycles_per_bit[i]
            threshold = ci * q_last / cfg.slot_length

            a1 = max(0.0, cfg.V * cfg.slot_length * up.phi_edge[i] / ci - lam) / (
                3.0 * cfg.V * cfg.slot_length * cfg.kappa_edge
            )
            a2 = max(0.0, cfg.slot_length * QE_total[i] / ci - lam) / (
                3.0 * cfg.V * cfg.slot_length * cfg.kappa_edge
            )
            cand1 = np.clip(clipped_sqrt(a1), 0.0, threshold)
            cand2 = max(threshold, clipped_sqrt(a2))

            def edge_objective(freq: float) -> float:
                mu = cfg.slot_length * freq / ci
                rec = max(0.0, q_last - mu)
                return (
                    cfg.V * cfg.slot_length * cfg.kappa_edge * freq**3
                    - cfg.slot_length * QE_total[i] * freq / ci
                    + (cfg.V * up.phi_edge[i] - QE_total[i]) * rec
                    + lam * freq
                )

            candidates = [cand1, cand2]
            values = [edge_objective(candidate) for candidate in candidates]
            return candidates[int(np.argmin(values))]

        lam_lo = 0.0
        weights = np.maximum(cfg.V * up.phi_edge, QE_total) * cfg.slot_length / up.cycles_per_bit
        lam_hi = np.max(weights)
        best = np.zeros(K)

        for _ in range(cfg.bisection_max_iter):
            lam = 0.5 * (lam_lo + lam_hi)
            current = np.array([best_f_for_user(i, lam) for i in range(K)])
            if current.sum() > cfg.edge_cpu_max_total:
                lam_lo = lam
            else:
                lam_hi = lam
                best = current
            if abs(lam_hi - lam_lo) < cfg.bisection_tol:
                best = current
                break
        return best


class ADDOSQPolicy(BasePolicy):
    """ADDO using a conventional single queue instead of age-layered queues.

    Unlike :class:`ADDOPolicy`, this policy does not inspect the oldest age
    layer or include the corresponding overdue-recourse terms.  The
    environment still maintains age layers so deadline metrics remain
    directly comparable with ADDO.
    """

    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        up = self.up
        QL_total = obs["QL_total"]
        QE_total = obs["QE_total"]
        channel = obs["channel"]

        fL = np.clip(
            np.sqrt(np.maximum(QL_total, 0.0) / (3.0 * cfg.V * cfg.kappa_local * up.cycles_per_bit)),
            0.0,
            up.local_cpu_max,
        )

        p = np.zeros(len(up.deadline))
        for i in range(len(up.deadline)):
            bandwidth = up.bandwidth[i]
            term = (
                (QL_total[i] - QE_total[i]) * bandwidth / (cfg.V * math.log(2.0))
                - cfg.noise_density * bandwidth / max(channel[i], 1e-12)
            )
            p[i] = np.clip(term, 0.0, up.tx_power_max[i])

        return {"fL": fL, "p": p, "fE": self._solve_edge_cpu(QE_total)}

    def _solve_edge_cpu(self, queue: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        up = self.up

        def allocation(lam: float) -> np.ndarray:
            numerator = np.maximum(cfg.slot_length * queue / up.cycles_per_bit - lam, 0.0)
            return np.sqrt(numerator / (3.0 * cfg.V * cfg.slot_length * cfg.kappa_edge))

        lam_lo = 0.0
        lam_hi = float(np.max(cfg.slot_length * queue / up.cycles_per_bit))
        best = allocation(lam_hi)

        for _ in range(cfg.bisection_max_iter):
            lam = 0.5 * (lam_lo + lam_hi)
            current = allocation(lam)
            if current.sum() > cfg.edge_cpu_max_total:
                lam_lo = lam
            else:
                lam_hi = lam
                best = current
            if abs(lam_hi - lam_lo) < cfg.bisection_tol:
                break
        return best


class DAEE(BasePolicy):
    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        up = self.up
        K = len(up.deadline)
        QL_total = obs["QL_total"]
        QE_total = obs["QE_total"]
        ZL = obs["ZL"]
        channel = obs["channel"]

        # Classical drift-plus-penalty: optimize current service cost
        # against aggregate queue reduction, without deadline overdue terms.
        fL = np.zeros(K)
        p = np.zeros(K)

        for i in range(K):
            ci = up.cycles_per_bit[i]
            fL[i] = np.clip(clipped_sqrt((QL_total[i] + ZL[i]) / (3.0 * cfg.V * cfg.kappa_local * ci)), 0.0, up.local_cpu_max[i])

            bandwidth = up.bandwidth[i]
            term = (
                (QL_total[i] + ZL[i] - QE_total[i]) * bandwidth / (cfg.V * math.log(2.0))
                - cfg.noise_density * bandwidth / max(channel[i], 1e-12)
            )
            p[i] = np.clip(term, 0.0, up.tx_power_max[i])

        return {"fL": fL, "p": p, "fE": self._solve_edge_cpu(obs)}

    def _solve_edge_cpu(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        cfg = self.cfg
        up = self.up
        QE_total = obs["QE_total"]
        ZE = obs["ZE"]
        K = len(up.deadline)

        def best_f_for_user(i: int, lam: float) -> float:
            ci = up.cycles_per_bit[i]
            queue_weight = max(QE_total[i] + ZE[i], 0.0)
            if queue_weight <= 0:
                return 0.0
            numerator = max(cfg.slot_length * queue_weight / ci - lam, 0.0)
            return np.clip(
                clipped_sqrt(numerator / (3.0 * cfg.V * cfg.slot_length * cfg.kappa_edge)),
                0.0,
                cfg.edge_cpu_max_total,
            )

        lam_lo = 0.0
        weights = cfg.slot_length * np.maximum(QE_total + ZE, 0.0) / up.cycles_per_bit
        lam_hi = float(np.max(weights))
        best = np.zeros(K)

        for _ in range(cfg.bisection_max_iter):
            lam = 0.5 * (lam_lo + lam_hi)
            current = np.array([best_f_for_user(i, lam) for i in range(K)])
            if current.sum() > cfg.edge_cpu_max_total:
                lam_lo = lam
            else:
                lam_hi = lam
                best = current
            if abs(lam_hi - lam_lo) < cfg.bisection_tol:
                best = current
                break
        return best


class LocalOnlyPolicy(BasePolicy):
    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        K = len(self.up.deadline)
        return {"fL": self.up.local_cpu_max.copy(), "p": np.zeros(K), "fE": np.zeros(K)}


class EdgeOnlyPolicy(BasePolicy):
    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        up = self.up
        K = len(up.deadline)
        ql_total = obs["QL_total"]

        p = np.where(ql_total > 0, up.tx_power_max, 0.0)
        weights = ql_total.copy()
        if np.sum(weights) <= 1e-12:
            fE = np.zeros(K)
        else:
            fE = cfg.edge_cpu_max_total * weights / np.sum(weights)
        return {"fL": np.zeros(K), "p": p, "fE": fE}


class EdgeFirstGreedyPolicy(BasePolicy):
    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        up = self.up
        K = len(up.deadline)
        QL_total = obs["QL_total"]
        QE_total = obs["QE_total"]

        p = np.where(QL_total > QE_total, up.tx_power_max, 0.2 * up.tx_power_max)
        fL = np.where(QL_total > 0, 0.3 * up.local_cpu_max, 0.0)
        weights = QE_total.copy()
        if np.sum(weights) <= 1e-12:
            fE = np.zeros(K)
        else:
            fE = cfg.edge_cpu_max_total * weights / np.sum(weights)
        return {"fL": fL, "p": p, "fE": fE}


class RandomAllocationPolicy(BasePolicy):
    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        K = len(self.up.deadline)
        fL = self.rng.uniform(0.0, self.up.local_cpu_max)
        p = self.rng.uniform(0.0, self.up.tx_power_max)

        if self.cfg.edge_cpu_max_total <= 0 or K == 0:
            fE = np.zeros(K)
        else:
            weights = self.rng.dirichlet(np.ones(K))
            fE = self.cfg.edge_cpu_max_total * weights

        return {"fL": fL, "p": p, "fE": fE}


POLICY_MAP = {
    "ADDO": ADDOPolicy,
    "ADDO-SQ": ADDOSQPolicy,
    "DAEE": DAEE,
    "LocalOnly": LocalOnlyPolicy,
    "EdgeOnly": EdgeOnlyPolicy,
    "EdgeFirstGreedy": EdgeFirstGreedyPolicy,
    "Random": RandomAllocationPolicy,
}
