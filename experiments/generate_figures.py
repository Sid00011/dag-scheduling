"""
Figure generation for the IEEE paper.
Produces all plots used in the results section.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'results.csv')

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
})

# Color palette (matches paper table)
COLORS = {
    'HEFT':     '#2C5F8A',   # strong blue
    'CPOP':     '#5BA85C',   # green
    'HEFT-LC':  '#D95F02',   # orange (our method)
    'Random':   '#AAAAAA',   # gray
}
MARKERS = {'HEFT': 'o', 'CPOP': 's', 'HEFT-LC': 'D', 'Random': '^'}
ORDER = ['Random', 'CPOP', 'HEFT', 'HEFT-LC']

os.makedirs(FIG_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Drop rows with inf/nan makespan
    df = df[np.isfinite(df['makespan']) & np.isfinite(df['slr'])]
    return df


# ── Figure 1: Makespan by n_tasks and CCR ─────────────────────────────────

def fig_makespan_vs_ntasks(df: pd.DataFrame):
    """Line plot: mean makespan vs number of tasks, one line per scheduler."""
    data = df[df['dag_type'] == 'random'].copy()
    grouped = data.groupby(['scheduler', 'n_tasks'])['makespan'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for sched in ORDER:
        sub = grouped[grouped['scheduler'] == sched]
        ax.plot(sub['n_tasks'], sub['makespan'],
                color=COLORS[sched], marker=MARKERS[sched],
                linewidth=1.8, markersize=6, label=sched,
                zorder=3 if sched == 'HEFT-LC' else 2)

    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Mean makespan (time units)')
    ax.set_title('Makespan vs. graph size (random DAGs, averaged over clusters)')
    ax.legend(framealpha=0.9, loc='upper left')
    ax.set_xticks([10, 20, 30, 50])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_makespan_vs_ntasks.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig1_makespan_vs_ntasks.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved fig1_makespan_vs_ntasks")


# ── Figure 2: Load imbalance comparison ───────────────────────────────────

def fig_imbalance_boxplot(df: pd.DataFrame):
    """Box plot: load imbalance distribution per scheduler across all configs."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: all DAG types
    data_all = [df[df['scheduler'] == s]['imbalance'].dropna().values for s in ORDER]
    bp = axes[0].boxplot(data_all, patch_artist=True, notch=False,
                         medianprops={'color': 'white', 'linewidth': 2})
    for patch, sched in zip(bp['boxes'], ORDER):
        patch.set_facecolor(COLORS[sched])
        patch.set_alpha(0.85)
    axes[0].set_xticklabels(ORDER)
    axes[0].set_ylabel('Load imbalance (CV)')
    axes[0].set_title('Load imbalance — all configurations')

    # Right: per cluster type
    cluster_types = df['cluster'].unique()
    x = np.arange(len(cluster_types))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for i, sched in enumerate(ORDER):
        means = [df[(df['scheduler'] == sched) & (df['cluster'] == c)]['imbalance'].mean()
                 for c in cluster_types]
        axes[1].bar(x + offsets[i] * width, means, width,
                    color=COLORS[sched], alpha=0.85, label=sched)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace('_', '\n') for c in cluster_types], fontsize=8)
    axes[1].set_ylabel('Mean load imbalance (CV)')
    axes[1].set_title('Load imbalance by cluster type')
    axes[1].legend(framealpha=0.9, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_imbalance.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig2_imbalance.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved fig2_imbalance")


# ── Figure 3: SLR vs CCR ──────────────────────────────────────────────────

def fig_slr_vs_ccr(df: pd.DataFrame):
    """Line plot: Schedule Length Ratio vs CCR for random DAGs."""
    data = df[(df['dag_type'] == 'random') & df['ccr'].notna()].copy()
    grouped = data.groupby(['scheduler', 'ccr'])['slr'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for sched in ORDER:
        sub = grouped[grouped['scheduler'] == sched]
        ax.plot(sub['ccr'], sub['slr'],
                color=COLORS[sched], marker=MARKERS[sched],
                linewidth=1.8, markersize=6, label=sched,
                zorder=3 if sched == 'HEFT-LC' else 2)

    ax.axhline(1.0, color='#999', linestyle='--', linewidth=1, label='Optimal (SLR=1)')
    ax.set_xlabel('CCR (communication-to-computation ratio)')
    ax.set_ylabel('Schedule Length Ratio (SLR)')
    ax.set_title('SLR vs. CCR — lower is better')
    ax.legend(framealpha=0.9)
    ax.set_xticks([0.1, 0.5, 1.0, 2.0])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_slr_vs_ccr.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig3_slr_vs_ccr.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved fig3_slr_vs_ccr")


# ── Figure 4: HEFT vs HEFT-LC makespan overhead ───────────────────────────

def fig_heft_lc_overhead(df: pd.DataFrame):
    """
    Scatter: for each DAG run, plot HEFT makespan vs HEFT-LC makespan.
    Points above the diagonal = HEFT-LC slightly worse;
    points below = HEFT-LC wins.
    """
    heft = df[df['scheduler'] == 'HEFT'][['dag_type', 'cluster', 'n_tasks', 'ccr', 'seed', 'makespan']].copy()
    lc   = df[df['scheduler'] == 'HEFT-LC'][['dag_type', 'cluster', 'n_tasks', 'ccr', 'seed', 'makespan', 'imbalance']].copy()
    heft.columns = ['dag_type', 'cluster', 'n_tasks', 'ccr', 'seed', 'heft_ms']
    lc.columns   = ['dag_type', 'cluster', 'n_tasks', 'ccr', 'seed', 'lc_ms', 'lc_imb']

    merged = heft.merge(lc, on=['dag_type', 'cluster', 'n_tasks', 'ccr', 'seed'])
    merged = merged[np.isfinite(merged['heft_ms']) & np.isfinite(merged['lc_ms'])]
    merged['overhead_pct'] = 100 * (merged['lc_ms'] - merged['heft_ms']) / merged['heft_ms']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: scatter
    lim = max(merged['heft_ms'].max(), merged['lc_ms'].max()) * 1.05
    axes[0].scatter(merged['heft_ms'], merged['lc_ms'],
                    alpha=0.3, s=10, color=COLORS['HEFT-LC'])
    axes[0].plot([0, lim], [0, lim], 'k--', linewidth=1, label='Equal makespan')
    axes[0].plot([0, lim], [0, lim * 1.05], color='#999', linewidth=1,
                 linestyle=':', label='+5% bound')
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_xlabel('HEFT makespan')
    axes[0].set_ylabel('HEFT-LC makespan')
    axes[0].set_title('Makespan: HEFT vs HEFT-LC')
    axes[0].legend(fontsize=8)

    # Right: histogram of overhead %
    ov = merged['overhead_pct'].clip(-20, 20)
    axes[1].hist(ov, bins=40, color=COLORS['HEFT-LC'], alpha=0.8, edgecolor='white')
    axes[1].axvline(0, color='k', linewidth=1.5, linestyle='--')
    axes[1].axvline(ov.mean(), color='#D95F02', linewidth=1.5,
                    label=f'Mean: {ov.mean():.1f}%')
    axes[1].set_xlabel('Makespan overhead vs HEFT (%)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of HEFT-LC overhead')
    axes[1].legend()

    pct_worse = (merged['overhead_pct'] > 1).mean() * 100
    pct_same  = (merged['overhead_pct'].abs() <= 1).mean() * 100
    pct_better = (merged['overhead_pct'] < -1).mean() * 100
    print(f"  HEFT-LC vs HEFT: {pct_better:.1f}% better, "
          f"{pct_same:.1f}% same (±1%), {pct_worse:.1f}% worse")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_heft_lc_overhead.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig4_heft_lc_overhead.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved fig4_heft_lc_overhead")


# ── Figure 5: Summary heatmap ─────────────────────────────────────────────

def fig_summary_heatmap(df: pd.DataFrame):
    """Heatmap: mean SLR per scheduler × cluster type."""
    pivot = df.groupby(['scheduler', 'cluster'])['slr'].mean().unstack()
    pivot = pivot.loc[ORDER]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0.5, vmax=2.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace('_', '\n') for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f'{pivot.values[i, j]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if pivot.values[i, j] > 1.5 else 'black')
    plt.colorbar(im, ax=ax, label='Mean SLR')
    ax.set_title('Schedule Length Ratio — lower is better')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_slr_heatmap.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig5_slr_heatmap.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved fig5_slr_heatmap")


# ── Figure 6: Speedup CDF ─────────────────────────────────────────────────

def fig_speedup_cdf(df: pd.DataFrame):
    """CDF of speedup values for each algorithm."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for sched in ORDER:
        vals = np.sort(df[df['scheduler'] == sched]['speedup'].dropna().values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, color=COLORS[sched], linewidth=2, label=sched)
    ax.set_xlabel('Speedup over sequential execution')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative distribution of speedup')
    ax.set_xlim(0, None)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig6_speedup_cdf.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig6_speedup_cdf.png'), bbox_inches='tight')
    plt.close(fig)
    print("Saved fig6_speedup_cdf")


# ── Summary table ─────────────────────────────────────────────────────────

def generate_summary_table(df: pd.DataFrame):
    """Produce the main results table (Table I in the paper)."""
    cols = ['makespan', 'speedup', 'efficiency', 'imbalance', 'slr']
    table = df.groupby('scheduler')[cols].agg(['mean', 'std']).round(3)
    table.index = pd.CategoricalIndex(table.index, categories=ORDER, ordered=True)
    table = table.sort_index()

    path = os.path.join(FIG_DIR, 'table1_summary.csv')
    table.to_csv(path)
    print(f"Saved table1_summary.csv")

    # Also print a clean LaTeX table
    latex_rows = []
    for sched in ORDER:
        sub = df[df['scheduler'] == sched]
        ms  = f"{sub['makespan'].mean():.1f} ± {sub['makespan'].std():.1f}"
        sp  = f"{sub['speedup'].mean():.2f}"
        imb = f"{sub['imbalance'].mean():.3f}"
        slr = f"{sub['slr'].mean():.3f}"
        marker = " *" if sched == 'HEFT-LC' else ""
        latex_rows.append(f"  {sched+marker:<12} & {ms:<22} & {sp:<6} & {imb:<7} & {slr}")

    print("\nTable I (for paper):")
    print("  Scheduler    | Makespan (mean±std)   | Speedup | Imbalance | SLR")
    print("  " + "-" * 65)
    for row in latex_rows:
        print(row)
    print("  (* proposed method)")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...")
    df = load_data()
    print(f"  {len(df)} rows, {df['scheduler'].nunique()} schedulers, "
          f"{df['dag_type'].nunique()} DAG types\n")

    print("Generating figures...")
    fig_makespan_vs_ntasks(df)
    fig_imbalance_boxplot(df)
    fig_slr_vs_ccr(df)
    fig_heft_lc_overhead(df)
    fig_summary_heatmap(df)
    fig_speedup_cdf(df)
    generate_summary_table(df)

    print(f"\nAll figures saved to {FIG_DIR}/")
