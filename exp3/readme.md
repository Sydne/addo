# Deterministic-Delay Edge Computing Experiments

This directory contains a time-slotted multi-user mobile edge computing (MEC) simulator for comparing deadline-aware task scheduling, computation offloading, and resource-allocation policies. The simulator reports local-computing, wireless-transmission, edge-computing, deadline-violation, queue-backlog, and task-delay metrics.

This document describes the code currently present in the workspace. The `results/` directory also contains files produced by older code versions. An algorithm name appearing in that directory does not necessarily mean that the algorithm is still available in the current implementation.

## 1. Environment Setup

Python 3.10 or later is recommended.

```powershell
pip install numpy pandas matplotlib openpyxl
```

The project currently has no `requirements.txt` or automated test suite. Before running a full experiment, you can execute a small smoke test that does not write result files:

```powershell
python -c "from mec_sim.models import SystemConfig; from mec_sim.simulation import Simulator; c=SystemConfig(num_users=4, num_slots=100); print(Simulator(c, 'ADDO').run()[0])"
```

## 2. Project Structure

```text
.
|-- main.py                         # Current default entry point: runs Experiment 1 only
|-- plot_vary_v.py                  # ADDO, ADDO-SQ, and DAEE comparisons versus V
|-- plot_vary_deadline.py            # Fixed-V comparison versus the deadline upper bound
|-- mec_sim/
|   |-- models.py                    # Configuration, user parameters, and metric dataclasses
|   |-- environment.py               # Arrivals, channels, age queues, and slot updates
|   |-- policies.py                  # Policy implementations and the POLICY_MAP registry
|   |-- simulation.py                # Single-simulation loop and metric aggregation
|   |-- experiments.py               # Experiments 1--3 and their runner functions
|   |-- plotting.py                  # Shared plotting helpers
|   `-- utils.py                     # Directory, JSON, and numerical helpers
`-- results/                          # Generated JSON, Excel, and PNG files
```

## 3. Current System Model

### 3.1 Queues and Deadlines

- Each user receives a fixed relative deadline `D_i` when the environment is initialized.
- By default, `heterogeneous_deadline=True`, so `D_i` is sampled uniformly from the inclusive interval `[deadline_min, deadline_max]`. All subsequent tasks from the same user share that user's `D_i`.
- Both the local queue `QL` and edge queue `QE` are divided into age layers indexed from `0` to `D_i-1`.
- Service follows an oldest-task-first order. For tasks from one user with a fixed relative deadline, this is equivalent to EDF.
- An arrival observed in slot `t` enters the local age-0 queue at the end of that slot and becomes serviceable in slot `t+1`.
- Local execution occurs before offloading. Data offloaded during the current slot becomes available for edge service in the next slot.
- Data already in the oldest local layer cannot be offloaded because it would reach the edge only in the next slot and could no longer satisfy its end-to-end deadline.
- Data that remains unfinished after service in its deadline slot is removed from the queue, counted as overdue/violation bits, and charged a linear penalty.

### 3.2 Communication and Computation Resources

- Per-user local CPU: `0 <= fL_i <= local_cpu_max`.
- Per-user transmit power: `0 <= p_i <= tx_power_max`.
- Each user has a fixed bandwidth value `bandwidth`; the current model has no shared cross-user bandwidth-allocation constraint.
- Edge CPU is shared: `sum_i fE_i <= edge_cpu_max_total`.
- Local and edge service amounts are calculated as `slot_length * frequency / cycles_per_bit`.
- Transmission rates use the Shannon formula. The channel power gain in each slot is sampled from the exponential distribution associated with Rayleigh block fading.

The environment does not automatically clip policy actions. Any new policy must enforce the power and CPU constraints above itself.

### 3.3 Costs

The operating cost in each slot has three components:

```text
local_cost = slot_length * kappa_local * fL^3
tx_cost    = slot_length * p
edge_cost  = slot_length * kappa_edge * fE^3
```

These costs are calculated from the allocated actions rather than actual resource utilization. Total cost is the sum of operating cost and the overdue penalty.

## 4. Default Configuration

Default values are defined by `SystemConfig` in `mec_sim/models.py`.

| Category | Parameter | Default | Meaning |
|---|---|---:|---|
| Simulation | `num_users` | `20` | Number of users |
| Simulation | `slot_length` | `0.01` s | Slot duration |
| Simulation | `num_slots` | `10000` | Total number of slots |
| Simulation | `warmup_slots` | `0` | Slots excluded from summary metrics |
| Simulation | `random_seed` | `42` | Environment random seed |
| Deadline | `deadline_min` | `2` | Lower bound for heterogeneous deadlines |
| Deadline | `deadline_max` | `10` | Deadline upper bound or homogeneous deadline |
| Deadline | `heterogeneous_deadline` | `True` | Whether deadlines are sampled per user |
| Arrivals | `arrival_mode` | `"uniform"` | `uniform` or `bursty` |
| Arrivals | `arrival_max_bits` | `3.0e4` bits | Uniform-arrival upper bound |
| Arrivals | `arrival_load_scale` | `1.0` | Arrival-load scaling factor |
| Computation | `cycles_per_bit_min/max` | `500 / 1000` | Per-user cycles-per-bit range |
| Computation | `local_cpu_max` | `1.0e9` Hz | Per-user local CPU limit |
| Computation | `edge_cpu_max_total` | `10.0e9` Hz | Total edge CPU limit |
| Communication | `tx_power_max` | `0.5` W | Per-user transmit-power limit |
| Communication | `bandwidth` | `1.0e6` Hz | Per-user bandwidth |
| Communication | `user_distance` | `150.0` m | Default user distance |
| Cost | `kappa_local/edge` | `1.0e-28` | Cubic CPU-cost coefficients |
| Penalty | `phi_local/edge` | `0.1` | Linear overdue-bit coefficients |
| Lyapunov | `persist_local/edge` | `2.0e4` | Persistent virtual-queue increments |
| Lyapunov | `V` | `1.0e11` | Drift-plus-penalty tradeoff parameter |
| Output | `output_dir` | `"results"` | Root result directory |

Additional notes:

- Bursty arrivals also use `burst_low_max_bits`, `burst_high_min_bits`, `burst_high_max_bits`, and `burst_probability`.
- When `heterogeneous_channel=True`, user distances are sampled from `0.8` to `1.2` times the default distance.
- The `channel_mode` field currently has no branching logic; the environment always uses exponentially distributed Rayleigh power gains.
- An old comment next to `tx_power_max` says “100 mW,” but the configured value `0.5 W` is actually `500 mW`. The simulation uses the numerical value `0.5`.

Example custom configuration:

```python
from mec_sim.experiments import ExperimentManager
from mec_sim.models import SystemConfig

cfg = SystemConfig(
    num_slots=500,
    random_seed=7,
    V=1e10,
    arrival_load_scale=0.8,
    output_dir="results_debug",
)
manager = ExperimentManager(cfg)
manager.experiment_2_v_tradeoff([1e8, 1e9, 1e10], policy="ADDO")
```

## 5. Currently Available Policies

The policy registry is `POLICY_MAP` near the end of `mec_sim/policies.py`. The following names can currently be passed to `Simulator`:

| Policy | Current implementation |
|---|---|
| `ADDO` | Uses aggregate queues and the oldest age layer to jointly select local CPU, transmit power, and shared edge CPU while explicitly accounting for the overdue penalty |
| `ADDO-SQ` | Bases resource decisions only on aggregate queues and does not use the oldest age layer or overdue-recourse term; the environment still uses age-layered queues and oldest-first service |
| `DAEE` | Conventional drift-plus-penalty using physical queues and persistent virtual queues `ZL/ZE` |
| `LocalOnly` | Always uses maximum local CPU and never offloads |
| `EdgeOnly` | Uses maximum transmit power when local backlog exists and allocates edge CPU in proportion to local backlog |
| `EdgeFirstGreedy` | Uses a fixed fraction of local CPU, chooses transmit power from the local/edge backlog relation, and allocates edge CPU in proportion to edge backlog |
| `Random` | Randomizes local CPU and transmit power and uses Dirichlet weights for edge CPU allocation |

Note that the current code does not register `TORA`, `EDF-Lyapunov`, `StandardLyapunov`, `ADDO2`, `EqualAllocation`, or `RandomAllocation`. These names may appear in historical JSON files, but they cannot be passed directly to the current `Simulator`.

## 6. Running Experiments

### 6.1 Current `main.py`

```powershell
python main.py
```

The current entry point runs all three experiments:

```python
run_all_experiments()
```

All other experiment calls are commented out. The default configuration uses 10,000 slots, and Experiment 1 runs several policies sequentially, so a full run can take considerable time. Use a smaller custom `SystemConfig` while debugging.

### 6.2 Running Individual Standard Experiments

```powershell
python -c "from mec_sim.experiments import run_experiment_1; run_experiment_1()"
python -c "from mec_sim.experiments import run_experiment_2; run_experiment_2(policy='ADDO')"
python -c "from mec_sim.experiments import run_experiment_3; run_experiment_3()"
```

`run_all_experiments()` runs Experiments 1--3 sequentially and is called by the current `main.py`:

```powershell
python -c "from mec_sim.experiments import run_all_experiments; run_all_experiments()"
```

### 6.3 Experiment Definitions

| Experiment | Default sweep | Default policies | Main outputs |
|---|---|---|---|
| Experiment 1: algorithm comparison | One default configuration | `ADDO`, `ADDO-SQ`, `DAEE`, `LocalOnly`, `EdgeOnly`, `Random` | Per-policy JSON, Excel, and three bar charts |
| Experiment 2: V tradeoff | 13 `V` values | One supplied policy per call; default is `ADDO` | One JSON file per V value |
| Experiment 3: deadline comparison | `[2, 3, 4, 5, 6, 8, 10]` | `ADDO`, `ADDO-SQ`, `DAEE`, `LocalOnly`, `EdgeOnly`, `Random` | Per-condition JSON and Excel |

In Experiment 3, the supplied `deadline` value is used as `deadline_max`; it is not a common fixed deadline for every user. Each point sets:

```python
heterogeneous_deadline = True
deadline_min = 2
deadline_max = deadline
```

To give every user the same fixed deadline, create `SystemConfig(heterogeneous_deadline=False, deadline_max=D)` and run the desired policy directly.

## 7. Result Directories and Filenames

```text
results/
|-- exp1_algorithm_comparison/
|-- exp2_v_tradeoff/
`-- exp3_deadline_comparison_fixed_v/
```

Typical filenames:

```text
exp1_algorithm_comparison/ADDO_summary.json
exp1_algorithm_comparison/experiment_1_summary.xlsx
exp2_v_tradeoff/ADDO_V_100000000000.0_summary.json
exp3_deadline_comparison_fixed_v/ADDO_V_100000000000.0_deadline_5.json
```

Experiments overwrite files with the same name without checking whether the seed, code version, or complete configuration matches. Apart from selected `V`, load, and deadline tokens, filenames contain no configuration hash. Use a new `output_dir` or back up old results before running final experiments.

## 8. Metric Definitions

| Field | Meaning |
|---|---|
| `avg_total_cost` | Average per-slot total cost: operating cost plus overdue penalty |
| `avg_operating_cost` | Average per-slot local-computing, transmission, and edge-computing cost |
| `avg_local_cost`, `avg_tx_cost`, `avg_edge_cost` | Components of operating cost |
| `avg_overdue_cost` | Average per-slot overdue penalty |
| `avg_total_backlog` | Average local-plus-edge queued bits after each slot update |
| `avg_local_backlog`, `avg_edge_backlog` | Backlog components |
| `avg_overdue_total_bits` | Average bits that expire unfinished per slot |
| `total_overdue_bits` | Total bits that expire unfinished during the aggregation window |
| `avg_offloaded_bits` | Average bits actually offloaded per slot |
| `offload_ratio` | Total actually offloaded bits divided by total arrival bits |
| `overdue_ratio` | Total expired unfinished bits divided by total arrival bits |
| `timeout_ratio` | Compatibility alias for `overdue_ratio` |
| `avg_sojourn_slots` | Bit-weighted departure delay; both completed and expired data are included |

`avg_sojourn_slots` is not averaged over a count of discrete jobs. The internal field `sojourn_jobs` actually accumulates bit weights.

When `warmup_slots > 0`, records from the warm-up period are excluded from the summary, but their queue evolution still affects the measured period.

Example JSON loading:

```python
import json

path = "results/exp2_v_tradeoff/ADDO_V_100000000000.0_summary.json"
with open(path, "r", encoding="utf-8") as file:
    summary = json.load(file)

print(summary["avg_operating_cost"])
print(summary["overdue_ratio"])
print(summary["avg_sojourn_slots"])
```

## 9. Plotting Prerequisites

The plotting scripts only read existing JSON files; they do not run missing experiments automatically. Their policy lists and x-axis values are hardcoded near the top of each script.

### `plot_addo_vary_v.py`

This script reads ADDO results for its `V_LIST`. The current list contains `8e10` and `1.8e11`, but those values are not included in the default list used by `run_experiment_2()`. Plotting directly from default Experiment 2 results may therefore raise `FileNotFoundError`. Generate data with the same list used by the plotting script:

```powershell
python -c "from mec_sim.experiments import run_experiment_2; run_experiment_2(policy='ADDO', v_values=[1e8,1e9,2e9,5e9,1e10,2e10,5e10,8e10,1e11,1.2e11,1.5e11,1.8e11,2e11])"
python plot_addo_vary_v.py
```

### `plot_addo_vs_daee.py`

This script requires JSON files for `ADDO`, `ADDO-SQ`, and `DAEE` at every specified V value. Run Experiment 2 separately for all three policies:

```powershell
python -c "from mec_sim.experiments import run_experiment_2; run_experiment_2(policy='ADDO')"
python -c "from mec_sim.experiments import run_experiment_2; run_experiment_2(policy='ADDO-SQ')"
python -c "from mec_sim.experiments import run_experiment_2; run_experiment_2(policy='DAEE')"
python plot_addo_vs_daee.py
```

### `plot_vary_deadline.py`

The current `POLICIES` list includes `TORA`, but `TORA` is not registered in the current `POLICY_MAP`. Existing files under `results/` include some historical TORA JSON results, so the script may still be useful for old data. To regenerate all results from scratch, first change `POLICIES` to policies that are currently registered and for which you will generate the complete deadline file grid, for example:

```python
POLICIES = ["ADDO", "ADDO-SQ", "DAEE", "LocalOnly", "EdgeOnly", "Random"]
```

Then run Experiment 3 with the same list before plotting:

```powershell
python -c "from mec_sim.experiments import run_experiment_3; run_experiment_3(deadlines=[2,3,4,5,6,8,10], policies=['ADDO','ADDO-SQ','DAEE','LocalOnly','EdgeOnly','Random'])"
python plot_vary_deadline.py
```

## 10. Fair Comparisons and Reproducibility

- Every `Simulator` creates an independent environment from the same `SystemConfig.random_seed`.
- Under an identical configuration, policies receive the same user parameters, arrival trace, and channel trace because policy actions do not change the number of environment RNG calls.
- A policy's own random-number generator is separate from the environment RNG. The `Random` policy therefore does not perturb later arrivals or channels.
- This fairness guarantee applies to comparisons between policies under the same configuration. Changing `deadline_max` changes random-number consumption during initialization and may consequently change later arrival/channel traces. Different x-axis points in Experiment 3 therefore do not use one fixed common trace.
- Standard experiments currently use only one seed and do not perform multi-seed repetitions, confidence intervals, or significance testing.

## 11. Adding a New Policy

1. In `mec_sim/policies.py`, create a class derived from `BasePolicy`.
2. Implement `act(self, obs)` and return three arrays named `fL`, `p`, and `fE`, each with length `num_users`.
3. Enforce nonnegative actions and the local-CPU, transmit-power, and total-edge-CPU constraints inside the policy.
4. Register a unique name in `POLICY_MAP`.
5. Add that name to the desired experiment's policy list. If plotting is required, also update the list in the corresponding plotting script.

Minimal structure:

```python
class MyPolicy(BasePolicy):
    def act(self, obs):
        return {"fL": fL, "p": p, "fE": fE}


POLICY_MAP = {
    # Keep the existing entries.
    "MyPolicy": MyPolicy,
}
```

A new policy should not modify `obs` or `self.cfg` in place, because doing so may compromise comparability within an experiment group.

## 12. Recommended Workflow

1. Inspect `SystemConfig` and the actual default policy list of the target experiment.
2. Use a separate `output_dir` and a small `num_slots` value for smoke testing.
3. Check action constraints, confirm that JSON metrics are finite, and verify that filenames match the plotting script's expectations.
4. Fix the seed and complete configuration, then run every policy being compared.
5. Confirm that the result-file grid is complete before running plotting scripts.
6. Preserve the code version, configuration, and result directory to prevent later runs from overwriting final results.
