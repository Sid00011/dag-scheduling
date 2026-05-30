"""
Heterogeneous cluster simulator for DAG scheduling research.
Models CPU, GPU, and edge nodes with distinct performance characteristics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np


class NodeType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    EDGE = "edge"


@dataclass
class ProcessorNode:
    """A processor in the heterogeneous cluster."""
    id: int
    node_type: NodeType
    speed_factor: float          # relative to baseline (1.0)
    memory_gb: float
    bandwidth_gbps: float        # inter-node bandwidth

    def __repr__(self):
        return f"P{self.id}[{self.node_type.value}, x{self.speed_factor:.1f}]"


@dataclass
class TaskAssignment:
    """Records when and where a task was scheduled."""
    task_id: int
    processor_id: int
    start_time: float
    finish_time: float

    @property
    def duration(self) -> float:
        return self.finish_time - self.start_time


class ClusterState:
    """
    Tracks the runtime state of the cluster during simulation.
    Maintains earliest-available times per processor and the schedule.
    """

    def __init__(self, processors: List[ProcessorNode]):
        self.processors = processors
        self.avail: Dict[int, float] = {p.id: 0.0 for p in processors}
        self.schedule: List[TaskAssignment] = []
        self.task_finish: Dict[int, float] = {}   # task_id -> finish time
        self.task_proc: Dict[int, int] = {}       # task_id -> processor_id
        self.proc_load: Dict[int, float] = {p.id: 0.0 for p in processors}

    def assign(self, task_id: int, proc_id: int, start: float, finish: float):
        self.avail[proc_id] = finish
        self.task_finish[task_id] = finish
        self.task_proc[task_id] = proc_id
        self.proc_load[proc_id] += (finish - start)
        self.schedule.append(TaskAssignment(task_id, proc_id, start, finish))

    def earliest_start(self, proc_id: int) -> float:
        return self.avail[proc_id]

    def makespan(self) -> float:
        return max(self.task_finish.values()) if self.task_finish else 0.0

    def load_imbalance(self) -> float:
        """
        Coefficient of variation of processor loads.
        0 = perfectly balanced, higher = more imbalanced.
        """
        loads = list(self.proc_load.values())
        mean = np.mean(loads)
        if mean == 0:
            return 0.0
        return np.std(loads) / mean

    def speedup(self, sequential_time: float) -> float:
        """Speedup over sequential execution on the fastest processor."""
        if self.makespan() == 0:
            return 0.0
        return sequential_time / self.makespan()

    def efficiency(self, sequential_time: float) -> float:
        return self.speedup(sequential_time) / len(self.processors)

    def schedule_length_ratio(self, critical_path: float) -> float:
        """SLR: makespan / critical path. Best possible = 1.0."""
        if critical_path == 0:
            return float('inf')
        return self.makespan() / critical_path


class ClusterFactory:
    """Builds standard cluster configurations used in experiments."""

    @staticmethod
    def homogeneous(n: int = 4) -> List[ProcessorNode]:
        """All identical CPU nodes — baseline cluster."""
        return [
            ProcessorNode(i, NodeType.CPU, speed_factor=1.0,
                          memory_gb=16, bandwidth_gbps=10)
            for i in range(n)
        ]

    @staticmethod
    def heterogeneous_cpu_gpu(n_cpu: int = 2, n_gpu: int = 2) -> List[ProcessorNode]:
        """
        Mixed CPU+GPU cluster.
        GPUs are 4x faster for compute but have lower memory bandwidth for
        task communication (modeled via lower bandwidth_gbps here).
        """
        procs = []
        for i in range(n_cpu):
            procs.append(ProcessorNode(
                i, NodeType.CPU, speed_factor=1.0,
                memory_gb=32, bandwidth_gbps=25
            ))
        for i in range(n_gpu):
            procs.append(ProcessorNode(
                n_cpu + i, NodeType.GPU, speed_factor=4.0,
                memory_gb=8, bandwidth_gbps=10
            ))
        return procs

    @staticmethod
    def edge_cloud(n_edge: int = 2, n_cloud: int = 2) -> List[ProcessorNode]:
        """
        Edge-cloud continuum: limited-resource edge nodes + powerful cloud.
        Models IoT scheduling scenarios.
        """
        procs = []
        for i in range(n_edge):
            procs.append(ProcessorNode(
                i, NodeType.EDGE, speed_factor=0.4,
                memory_gb=2, bandwidth_gbps=1
            ))
        for i in range(n_cloud):
            procs.append(ProcessorNode(
                n_edge + i, NodeType.CPU, speed_factor=2.0,
                memory_gb=64, bandwidth_gbps=40
            ))
        return procs

    @staticmethod
    def highly_heterogeneous(n: int = 4) -> List[ProcessorNode]:
        """
        Cluster used in original HEFT paper for comparison.
        Wide variation in speed factors.
        """
        speed_factors = [1.0, 2.5, 0.5, 3.0][:n]
        return [
            ProcessorNode(i, NodeType.CPU, speed_factor=speed_factors[i],
                          memory_gb=16, bandwidth_gbps=10)
            for i in range(n)
        ]


def effective_comp_cost(task_comp_costs: Dict[int, float],
                        processor: ProcessorNode) -> float:
    """
    Compute adjusted computation cost for a task on a given processor.
    The comp_cost already encodes heterogeneity via the task model,
    but we optionally apply speed_factor for cluster-driven variation.
    """
    base = task_comp_costs.get(processor.id, list(task_comp_costs.values())[0])
    return base / processor.speed_factor


def effective_comm_cost(comm_cost: float,
                        src_proc: ProcessorNode,
                        dst_proc: ProcessorNode) -> float:
    """
    Communication cost between two processors.
    Zero if same processor. Otherwise scaled by bandwidth.
    Reference bandwidth = 10 Gbps.
    """
    if src_proc.id == dst_proc.id:
        return 0.0
    ref_bandwidth = 10.0
    bw = min(src_proc.bandwidth_gbps, dst_proc.bandwidth_gbps)
    return comm_cost * (ref_bandwidth / bw)


if __name__ == "__main__":
    cluster = ClusterFactory.heterogeneous_cpu_gpu(n_cpu=2, n_gpu=2)
    print("Cluster:", cluster)
    state = ClusterState(cluster)
    state.assign(0, 0, 0, 5)
    state.assign(1, 1, 2, 8)
    state.assign(2, 0, 5, 11)
    print(f"Makespan: {state.makespan()}")
    print(f"Load imbalance: {state.load_imbalance():.3f}")
