# DAG Scheduling: HEFT-LC

**Scheduling heuristics for DAG task graphs in heterogeneous distributed systems**

Research project — M1 Informatique, Université Claude Bernard Lyon 1

## Summary

This repository implements and evaluates four DAG scheduling algorithms on heterogeneous processor clusters:

| Algorithm | Description |
|-----------|-------------|
| **HEFT** | Heterogeneous Earliest Finish Time (Topcuoglu et al., 1999) |
| **CPOP** | Critical Path on a Processor (Topcuoglu et al., 1999) |
| **HEFT-LC** ⭐ | *Our contribution* — HEFT with load-aware tie-breaking |
| Random | Random assignment baseline |

## Key Result

HEFT-LC reduces **load imbalance by 20%** vs HEFT (0.571 vs 0.714 CV, p < 0.001) with a bounded makespan overhead of ≤ε = 5%.

## Structure

```
src/
  dag_generator.py   — Random, layered, and benchmark DAG generation
  cluster.py         — Heterogeneous cluster model (CPU/GPU/edge)
  schedulers.py      — HEFT, CPOP, HEFT-LC, Random implementations
experiments/
  run_experiments.py — Full benchmark suite (2,720 DAGs × 4 clusters)
  generate_figures.py — Publication-quality figures
paper/
  paper_heft_lc.tex  — IEEE-format research paper (LaTeX)
data/
  results.csv        — Raw experimental results
figures/             — All generated plots (PDF + PNG)
```

## Quick Start

```bash
pip install networkx matplotlib numpy scipy pandas seaborn
python src/schedulers.py                        # single run demo
python experiments/run_experiments.py           # full benchmark (saves data/results.csv)
python experiments/generate_figures.py          # generate all figures
```

## Citation

```
Mezaourou, S. (2026). Load-Aware Tie-breaking for HEFT Scheduling in
Heterogeneous Clusters. Research report, Université Lyon 1.
```
