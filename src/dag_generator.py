"""
DAG Generator for heterogeneous scheduling research.
Generates synthetic task graphs with realistic computation/communication costs.
"""

import random
import math
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class TaskNode:
    """A node in the task DAG."""
    id: int
    comp_costs: Dict[int, float]   # processor_id -> computation cost
    label: str = ""

    def avg_comp(self) -> float:
        return np.mean(list(self.comp_costs.values()))


@dataclass
class DAGEdge:
    """A directed edge (dependency) with communication cost."""
    src: int
    dst: int
    comm_cost: float   # data transfer cost (0 if same processor)


class DAGGraph:
    """
    Directed Acyclic Graph of tasks for heterogeneous scheduling.
    Follows the standard formulation from Topcuoglu et al. (1999).
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.tasks: Dict[int, TaskNode] = {}
        self.edges: Dict[Tuple[int, int], DAGEdge] = {}
        self.n_processors: int = 0

    def add_task(self, task: TaskNode):
        self.tasks[task.id] = task
        self.G.add_node(task.id)

    def add_dependency(self, edge: DAGEdge):
        self.edges[(edge.src, edge.dst)] = edge
        self.G.add_edge(edge.src, edge.dst, weight=edge.comm_cost)

    def entry_tasks(self) -> List[int]:
        return [n for n in self.G.nodes if self.G.in_degree(n) == 0]

    def exit_tasks(self) -> List[int]:
        return [n for n in self.G.nodes if self.G.out_degree(n) == 0]

    def topological_order(self) -> List[int]:
        return list(nx.topological_sort(self.G))

    def comm_cost(self, src: int, dst: int) -> float:
        return self.edges.get((src, dst), DAGEdge(src, dst, 0)).comm_cost

    def n_tasks(self) -> int:
        return len(self.tasks)

    def critical_path_length(self) -> float:
        """Compute the critical path length using average computation costs."""
        longest = nx.dag_longest_path_length(self.G, weight='weight')
        # Add average computation of nodes on critical path
        path = nx.dag_longest_path(self.G, weight='weight')
        node_costs = sum(self.tasks[n].avg_comp() for n in path)
        return node_costs + longest


class DAGGenerator:
    """
    Generates random DAGs following the model used in HEFT/CPOP literature.

    Parameters mirror those in Topcuoglu et al. (1999) and
    Canon & Jeannot (2008) for reproducibility.
    """

    def __init__(self, n_processors: int = 4, seed: int = 42):
        self.n_processors = n_processors
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _gen_comp_costs(self, mean: float, ccr_hint: float = 0.5) -> Dict[int, float]:
        """
        Generate computation costs per processor.
        Heterogeneity factor beta drawn from [0, 2*mean] so avg stays at mean.
        """
        costs = {}
        for p in range(self.n_processors):
            # Uniform heterogeneity in [mean*(1-beta), mean*(1+beta)]
            beta = 0.5
            costs[p] = max(1.0, self.rng.uniform(mean * (1 - beta), mean * (1 + beta)))
        return costs

    def random_dag(
        self,
        n_tasks: int = 20,
        edge_density: float = 0.3,
        ccr: float = 0.5,
        mean_comp: float = 10.0,
        label: str = "random"
    ) -> DAGGraph:
        """
        Generate a random DAG.

        Args:
            n_tasks:       number of tasks (nodes)
            edge_density:  probability of edge between two tasks (0..1)
            ccr:           communication-to-computation ratio
            mean_comp:     mean computation cost
            label:         graph identifier
        """
        dag = DAGGraph()
        dag.n_processors = self.n_processors

        # Create tasks
        for i in range(n_tasks):
            task = TaskNode(
                id=i,
                comp_costs=self._gen_comp_costs(mean_comp),
                label=f"T{i}"
            )
            dag.add_task(task)

        # Create edges (only forward edges to keep DAG property)
        mean_comm = ccr * mean_comp
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                if self.rng.random() < edge_density:
                    comm = max(0.1, self.rng.gauss(mean_comm, mean_comm * 0.3))
                    dag.add_dependency(DAGEdge(src=i, dst=j, comm_cost=comm))

        # Ensure connectivity: every node reachable from at least one entry
        isolated = [n for n in dag.G.nodes if dag.G.in_degree(n) == 0
                    and dag.G.out_degree(n) == 0]
        for n in isolated:
            target = self.rng.randint(0, n_tasks - 1)
            if target != n:
                comm = max(0.1, self.rng.gauss(mean_comm, mean_comm * 0.3))
                if n < target:
                    dag.add_dependency(DAGEdge(src=n, dst=target, comm_cost=comm))
                else:
                    dag.add_dependency(DAGEdge(src=target, dst=n, comm_cost=comm))

        # Verify DAG property
        assert nx.is_directed_acyclic_graph(dag.G), "Generated graph has cycles!"
        return dag

    def laplacian_dag(self, n_tasks: int = 30, ccr: float = 1.0) -> DAGGraph:
        """
        Layered DAG: tasks organized in layers, each connected to some in next layer.
        More structured than random; models scientific workflows.
        """
        dag = DAGGraph()
        dag.n_processors = self.n_processors
        mean_comp = 10.0
        mean_comm = ccr * mean_comp

        n_layers = max(3, int(math.sqrt(n_tasks)))
        layer_sizes = [max(1, n_tasks // n_layers)] * n_layers
        remainder = n_tasks - sum(layer_sizes)
        for i in range(remainder):
            layer_sizes[i % n_layers] += 1

        task_id = 0
        layers: List[List[int]] = []
        for layer in layer_sizes:
            current = []
            for _ in range(layer):
                t = TaskNode(
                    id=task_id,
                    comp_costs=self._gen_comp_costs(mean_comp),
                    label=f"T{task_id}"
                )
                dag.add_task(t)
                current.append(task_id)
                task_id += 1
            layers.append(current)

        for li in range(len(layers) - 1):
            for src in layers[li]:
                # Each task connects to 1..3 tasks in next layer
                n_connections = self.rng.randint(1, min(3, len(layers[li + 1])))
                targets = self.rng.sample(layers[li + 1], n_connections)
                for dst in targets:
                    comm = max(0.1, self.rng.gauss(mean_comm, mean_comm * 0.2))
                    dag.add_dependency(DAGEdge(src=src, dst=dst, comm_cost=comm))

        assert nx.is_directed_acyclic_graph(dag.G)
        return dag

    def benchmark_epigenomics(self, scale: float = 1.0) -> DAGGraph:
        """
        Simplified Epigenomics workflow (widely used in scheduling benchmarks).
        Models a bioinformatics pipeline with fan-out/fan-in structure.
        """
        dag = DAGGraph()
        dag.n_processors = self.n_processors

        # Structure: 1 entry -> N map tasks -> 1 merge -> N reduce -> 1 exit
        N = 4
        tasks_def = []
        # Entry
        tasks_def.append((0, 5.0 * scale))
        # Map tasks
        for i in range(N):
            tasks_def.append((1 + i, 20.0 * scale))
        # Merge
        tasks_def.append((1 + N, 8.0 * scale))
        # Reduce tasks
        for i in range(N):
            tasks_def.append((2 + N + i, 15.0 * scale))
        # Exit
        tasks_def.append((2 + 2 * N, 5.0 * scale))

        for tid, mean in tasks_def:
            dag.add_task(TaskNode(
                id=tid,
                comp_costs=self._gen_comp_costs(mean),
                label=f"T{tid}"
            ))

        comm = 3.0 * scale
        # Entry -> maps
        for i in range(N):
            dag.add_dependency(DAGEdge(0, 1 + i, comm))
        # Maps -> merge
        for i in range(N):
            dag.add_dependency(DAGEdge(1 + i, 1 + N, comm * 2))
        # Merge -> reduces
        for i in range(N):
            dag.add_dependency(DAGEdge(1 + N, 2 + N + i, comm))
        # Reduces -> exit
        for i in range(N):
            dag.add_dependency(DAGEdge(2 + N + i, 2 + 2 * N, comm * 0.5))

        assert nx.is_directed_acyclic_graph(dag.G)
        return dag


if __name__ == "__main__":
    gen = DAGGenerator(n_processors=4, seed=0)
    dag = gen.random_dag(n_tasks=10, edge_density=0.4, ccr=0.5)
    print(f"Random DAG: {dag.n_tasks()} tasks, {len(dag.edges)} edges")
    print(f"Entry tasks: {dag.entry_tasks()}")
    print(f"Exit tasks:  {dag.exit_tasks()}")
    print(f"Critical path length: {dag.critical_path_length():.2f}")
