"""
Experiment runner for DAG scheduling benchmark.
Runs all schedulers across multiple configurations and collects metrics.
"""

import sys
import os
import json
import time
import itertools
from typing import List, Dict, Any
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dag_generator import DAGGenerator, DAGGraph
from cluster import ClusterFactory, ProcessorNode
from schedulers import HEFT, CPOP, HEFT_LC, RandomScheduler


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

SCHEDULERS = [HEFT(), CPOP(), HEFT_LC(epsilon=0.05), RandomScheduler(seed=0)]
SCHEDULER_NAMES = [s.name for s in SCHEDULERS]

CLUSTER_CONFIGS = {
    "homogeneous":        ClusterFactory.homogeneous(4),
    "cpu_gpu":            ClusterFactory.heterogeneous_cpu_gpu(2, 2),
    "edge_cloud":         ClusterFactory.edge_cloud(2, 2),
    "highly_hetero":      ClusterFactory.highly_heterogeneous(4),
}

N_TASKS_RANGE = [10, 20, 30, 50]
CCR_RANGE = [0.1, 0.5, 1.0, 2.0]          # communication-to-computation ratio
EDGE_DENSITY_RANGE = [0.2, 0.4, 0.6]
N_SEEDS = 10                                # repetitions per config


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment(dag: DAGGraph,
                   processors: List[ProcessorNode],
                   dag_label: str,
                   cluster_label: str,
                   config: Dict[str, Any]) -> List[Dict]:
    """Run all schedulers on one DAG and return rows for the results DataFrame."""
    seq_time = sum(t.avg_comp() for t in dag.tasks.values())
    cp = dag.critical_path_length()
    rows = []

    for sched in SCHEDULERS:
        t0 = time.perf_counter()
        state = sched.schedule(dag, processors)
        elapsed = time.perf_counter() - t0

        ms = state.makespan()
        rows.append({
            "scheduler":     sched.name,
            "dag_type":      dag_label,
            "cluster":       cluster_label,
            "n_tasks":       dag.n_tasks(),
            "n_edges":       len(dag.edges),
            "ccr":           config.get("ccr", None),
            "edge_density":  config.get("edge_density", None),
            "seed":          config.get("seed", None),
            "makespan":      ms,
            "speedup":       state.speedup(seq_time),
            "efficiency":    state.efficiency(seq_time),
            "imbalance":     state.load_imbalance(),
            "slr":           state.schedule_length_ratio(cp),
            "runtime_ms":    elapsed * 1000,
            "seq_time":      seq_time,
            "cp_length":     cp,
        })
    return rows


def run_all_experiments(verbose: bool = True) -> pd.DataFrame:
    rows = []
    total = len(N_TASKS_RANGE) * len(CCR_RANGE) * len(EDGE_DENSITY_RANGE) * N_SEEDS * len(CLUSTER_CONFIGS)
    done = 0

    if verbose:
        print(f"Running {total} DAG configurations × {len(SCHEDULERS)} schedulers...")

    # 1. Random DAGs
    for n_tasks, ccr, density, seed, (cluster_name, procs) in itertools.product(
            N_TASKS_RANGE, CCR_RANGE, EDGE_DENSITY_RANGE, range(N_SEEDS), CLUSTER_CONFIGS.items()):

        gen = DAGGenerator(n_processors=len(procs), seed=seed)
        dag = gen.random_dag(n_tasks=n_tasks, edge_density=density, ccr=ccr)
        config = {"ccr": ccr, "edge_density": density, "seed": seed, "n_tasks": n_tasks}
        rows.extend(run_experiment(dag, procs, "random", cluster_name, config))

        done += 1
        if verbose and done % 50 == 0:
            print(f"  {done}/{total} ({100*done//total}%)")

    # 2. Layered DAGs (scientific workflow model)
    for n_tasks, ccr, seed, (cluster_name, procs) in itertools.product(
            N_TASKS_RANGE, CCR_RANGE, range(N_SEEDS), CLUSTER_CONFIGS.items()):

        gen = DAGGenerator(n_processors=len(procs), seed=seed + 1000)
        dag = gen.laplacian_dag(n_tasks=n_tasks, ccr=ccr)
        config = {"ccr": ccr, "edge_density": None, "seed": seed, "n_tasks": n_tasks}
        rows.extend(run_experiment(dag, procs, "layered", cluster_name, config))

    # 3. Epigenomics benchmark (fixed structure, vary scale)
    for scale, seed, (cluster_name, procs) in itertools.product(
            [0.5, 1.0, 2.0, 5.0], range(N_SEEDS), CLUSTER_CONFIGS.items()):

        gen = DAGGenerator(n_processors=len(procs), seed=seed + 2000)
        dag = gen.benchmark_epigenomics(scale=scale)
        config = {"ccr": None, "edge_density": None, "seed": seed, "n_tasks": dag.n_tasks()}
        rows.extend(run_experiment(dag, procs, "epigenomics", cluster_name, config))

    df = pd.DataFrame(rows)
    if verbose:
        print(f"\nDone. {len(df)} result rows ({len(df)//len(SCHEDULERS)} unique DAG runs)")
    return df


if __name__ == "__main__":
    df = run_all_experiments(verbose=True)
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'results.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    print("\n--- Summary by scheduler ---")
    summary = df.groupby('scheduler')[['makespan', 'speedup', 'imbalance', 'slr']].mean().round(3)
    print(summary.to_string())
