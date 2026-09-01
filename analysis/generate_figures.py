"""Generate paper figures and tables from saved result artifacts only."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_FORMATS = ("png", "pdf", "svg")

def setup_ieee_style(column: str = "single") -> Tuple[float, float]:
    width = 3.5 if column == "single" else 7.16
    height = width * 0.62

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    return width, height


def ensure_dirs(base_out: Path) -> Dict[str, Path]:
    out = {
        "fig": base_out / "figures",
        "tbl": base_out / "tables",
    }
    out["fig"].mkdir(parents=True, exist_ok=True)
    out["tbl"].mkdir(parents=True, exist_ok=True)
    return out


def save_fig(fig: plt.Figure, out_path_stem: Path) -> None:
    for ext in OUTPUT_FORMATS:
        fig.savefig(out_path_stem.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def write_table(df: pd.DataFrame, stem: Path, float_fmt: str = "%.4f") -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    latex = df.to_latex(index=False, float_format=lambda x: float_fmt % x)
    stem.with_suffix(".tex").write_text(latex, encoding="utf-8")


def normalize_algo_name(name: str) -> str:
    n = str(name).strip().lower()
    mapping = {
        "hat": "HAT",
        "hagfish": "HAT",
        "hyperband": "Hyperband",
        "sha": "SHA",
        "asha": "ASHA",
        "tpe": "TPE",
        "bohb": "BOHB",
        "dehb": "DEHB",
        "dehb_style": "DEHB-style",
        "random_search": "Random Search",
        "simulatedannealing": "Simulated Annealing",
    }
    return mapping.get(n, str(name))


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


HPOBENCH_ALGO_ORDER = ["HAT", "Hyperband", "SHA", "TPE"]


def preferred_order(values: List[str], preferred: List[str]) -> List[str]:
    seen = {str(v) for v in values}
    ordered = [name for name in preferred if name in seen]
    ordered.extend(sorted(seen - set(preferred)))
    return ordered


def build_anytime_from_results_merged(path: Path, max_points_per_algo: int = 220) -> Optional[pd.DataFrame]:
    df = safe_read_csv(path)
    if df is None or df.empty:
        return None

    required = {"algorithm", "cumulative_simulated_cost", "best_validation_error"}
    if not required.issubset(df.columns):
        return None

    d = df.copy()
    d["algorithm"] = d["algorithm"].map(normalize_algo_name)
    d["cumulative_simulated_cost"] = pd.to_numeric(d["cumulative_simulated_cost"], errors="coerce")
    d["best_validation_error"] = pd.to_numeric(d["best_validation_error"], errors="coerce")
    d = d.dropna(subset=["algorithm", "cumulative_simulated_cost", "best_validation_error"])
    d = d[d["cumulative_simulated_cost"] > 0]
    if d.empty:
        return None

    if "seed" in d.columns:
        rows: List[pd.DataFrame] = []
        for algo, g_algo in d.groupby("algorithm"):
            curves: List[Tuple[np.ndarray, np.ndarray]] = []

            # A seed identifier is reused on every dataset.  Treat each
            # dataset-seed pair as an independent trajectory so curves from
            # different tasks are never concatenated into a synthetic run.
            curve_keys = ["seed"]
            if "dataset" in g_algo.columns:
                curve_keys = ["dataset", "seed"]

            for _, g_seed in g_algo.groupby(curve_keys):
                g = g_seed.sort_values("cumulative_simulated_cost")
                g = g.groupby("cumulative_simulated_cost", as_index=False)["best_validation_error"].min()
                x = g["cumulative_simulated_cost"].to_numpy(dtype=float)
                y = g["best_validation_error"].to_numpy(dtype=float)

                if x.size < 2 or x[-1] <= x[0]:
                    continue
                curves.append((x, y))

            if not curves:
                continue

            min_cost = max(min(curve[0][0] for curve in curves), 1e-12)
            max_cost = min(curve[0][-1] for curve in curves)
            if max_cost <= min_cost:
                continue

            grid = np.geomspace(min_cost, max_cost, num=max_points_per_algo)
            interp_curves = np.vstack([
                np.interp(grid, x, y, left=y[0], right=y[-1]) for x, y in curves
            ])

            rows.append(
                pd.DataFrame(
                    {
                        "algorithm": algo,
                        "cumulative_cost": grid,
                        "mean_best_validation_error": interp_curves.mean(axis=0),
                        "std_best_validation_error": interp_curves.std(axis=0, ddof=0),
                    }
                )
            )

        if rows:
            return pd.concat(rows, ignore_index=True)
        return None

    agg = (
        d.groupby(["algorithm", "cumulative_simulated_cost"], as_index=False)
        .agg(
            mean_best_validation_error=("best_validation_error", "mean"),
            std_best_validation_error=("best_validation_error", "std"),
        )
        .rename(columns={"cumulative_simulated_cost": "cumulative_cost"})
    )
    agg["std_best_validation_error"] = agg["std_best_validation_error"].fillna(0.0)

    sampled = []
    for _, grp in agg.groupby("algorithm"):
        g = grp.sort_values("cumulative_cost")
        if len(g) > max_points_per_algo:
            idx = np.unique(np.linspace(0, len(g) - 1, max_points_per_algo).astype(int))
            g = g.iloc[idx]
        sampled.append(g)

    if not sampled:
        return None
    return pd.concat(sampled, ignore_index=True)


def build_hpobench_figures_tables(root: Path, out_dirs: Dict[str, Path], column: str) -> List[str]:
    created: List[str] = []
    fig_w, fig_h = setup_ieee_style(column)

    base = root / "results" / "hpobench"
    stats_dir = base / "statistics_auc"

    run_level = safe_read_csv(stats_dir / "run_level_metrics.csv")
    dataset_summary = safe_read_csv(stats_dir / "dataset_algorithm_summary.csv")
    global_rank = safe_read_csv(stats_dir / "global_rankings.csv")
    rankings = safe_read_csv(stats_dir / "dataset_rankings.csv")
    anytime: Optional[pd.DataFrame] = None
    anytime_sources: List[Tuple[Path, str]] = [
        (base / "results_merged_labeled.csv", "merged"),
    ]

    for source_path, source_kind in anytime_sources:
        if source_kind == "merged":
            candidate = build_anytime_from_results_merged(source_path)
        else:
            candidate = safe_read_csv(source_path)

        if candidate is not None and not candidate.empty:
            anytime = candidate
            break

    if run_level is not None and not run_level.empty:
        run_level = run_level.copy()
        run_level["algorithm"] = run_level["algorithm"].map(normalize_algo_name)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        sns.boxplot(data=run_level, x="algorithm", y="auc_log_cost", hue="algorithm", legend=False, ax=ax, palette="colorblind")
        sns.stripplot(data=run_level, x="algorithm", y="auc_log_cost", ax=ax, color="black", alpha=0.35, size=2)
        ax.set_title("HPOBench: AUC Log-Cost Distribution")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("AUC (log cost)")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, out_dirs["fig"] / "hpobench_auc_boxplot_ieee")
        created.append("hpobench_auc_boxplot_ieee")

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        sns.violinplot(data=run_level, x="algorithm", y="auc_log_cost", hue="algorithm", legend=False, inner="quartile", ax=ax, palette="muted")
        ax.set_title("HPOBench: AUC Log-Cost Violin")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("AUC (log cost)")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, out_dirs["fig"] / "hpobench_auc_violin_ieee")
        created.append("hpobench_auc_violin_ieee")

    if anytime is not None and not anytime.empty:
        anytime = anytime.copy()
        anytime["algorithm"] = anytime["algorithm"].map(normalize_algo_name)

        if "cumulative_simulated_cost" in anytime.columns and "cumulative_cost" not in anytime.columns:
            anytime = anytime.rename(columns={"cumulative_simulated_cost": "cumulative_cost"})
        if "best_validation_error" in anytime.columns and "mean_best_validation_error" not in anytime.columns:
            anytime = anytime.rename(columns={"best_validation_error": "mean_best_validation_error"})
        if "std_best_validation_error" not in anytime.columns:
            anytime["std_best_validation_error"] = 0.0

        anytime["cumulative_cost"] = pd.to_numeric(anytime["cumulative_cost"], errors="coerce")
        anytime["mean_best_validation_error"] = pd.to_numeric(anytime["mean_best_validation_error"], errors="coerce")
        anytime["std_best_validation_error"] = pd.to_numeric(anytime["std_best_validation_error"], errors="coerce").fillna(0.0)
        anytime = anytime.dropna(subset=["algorithm", "cumulative_cost", "mean_best_validation_error"])
        anytime = anytime[anytime["cumulative_cost"] > 0]

        algo_order = preferred_order(anytime["algorithm"].unique().tolist(), HPOBENCH_ALGO_ORDER)
        palette = sns.color_palette("colorblind", n_colors=max(len(algo_order), 1))
        color_map = {algo: palette[i] for i, algo in enumerate(algo_order)}

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        for algo in algo_order:
            g = anytime[anytime["algorithm"] == algo].sort_values("cumulative_cost")
            if g.empty:
                continue

            lower = np.maximum(g["mean_best_validation_error"] - g["std_best_validation_error"], 0.0)
            upper = g["mean_best_validation_error"] + g["std_best_validation_error"]

            ax.plot(
                g["cumulative_cost"],
                g["mean_best_validation_error"],
                label=algo,
                linewidth=1.6,
                color=color_map[algo],
            )
            ax.fill_between(
                g["cumulative_cost"],
                lower,
                upper,
                alpha=0.10,
                color=color_map[algo],
            )
        ax.set_xscale("log")
        ax.set_title("HPOBench Anytime Performance")
        ax.set_xlabel("Cumulative cost (log scale)")
        ax.set_ylabel("Mean best validation error")
        ax.legend(loc="best", ncol=2 if len(algo_order) > 3 else 1)
        save_fig(fig, out_dirs["fig"] / "hpobench_anytime_curve_ieee")
        created.append("hpobench_anytime_curve_ieee")

    if global_rank is not None and not global_rank.empty:
        gr = global_rank.copy()
        gr["algorithm"] = gr["algorithm"].map(normalize_algo_name)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        sns.barplot(data=gr, x="algorithm", y="avg_rank", hue="algorithm", legend=False, ax=ax, palette="colorblind")
        ax.set_title("HPOBench Global Average Rank")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Average rank (lower is better)")
        ax.tick_params(axis="x", rotation=20)
        save_fig(fig, out_dirs["fig"] / "hpobench_global_rank_bar_ieee")
        created.append("hpobench_global_rank_bar_ieee")

        write_table(gr.sort_values("avg_rank"), out_dirs["tbl"] / "table_hpobench_global_rankings")
        created.append("table_hpobench_global_rankings")

    if rankings is not None and not rankings.empty:
        rr = rankings.copy()
        rr["algorithm"] = rr["algorithm"].map(normalize_algo_name)
        rr = rr.pivot_table(index="dataset", columns="algorithm", values="rank", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(fig_w, max(fig_h, 2.8)), constrained_layout=True)
        sns.heatmap(rr, annot=True, fmt=".2f", cmap="YlGnBu_r", cbar_kws={"label": "Rank"}, ax=ax)
        ax.set_title("HPOBench Dataset-wise Ranking Heatmap")
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Dataset")
        save_fig(fig, out_dirs["fig"] / "hpobench_dataset_rank_heatmap_ieee")
        created.append("hpobench_dataset_rank_heatmap_ieee")

        write_table(rr.reset_index(), out_dirs["tbl"] / "table_hpobench_dataset_rankings")
        created.append("table_hpobench_dataset_rankings")

    if dataset_summary is not None and not dataset_summary.empty:
        ds = dataset_summary.copy()
        ds["algorithm"] = ds["algorithm"].map(normalize_algo_name)
        write_table(ds, out_dirs["tbl"] / "table_hpobench_dataset_algorithm_summary")
        created.append("table_hpobench_dataset_algorithm_summary")

    return created


def build_lcbench_figures_tables(root: Path, out_dirs: Dict[str, Path], column: str) -> List[str]:
    created: List[str] = []
    fig_w, fig_h = setup_ieee_style(column)

    final_runs = safe_read_csv(root / "results" / "lcbench" / "final_runs.csv")
    all_results = None
    summary = None
    if final_runs is not None and not final_runs.empty:
        summary = (
            final_runs.groupby(["algorithm", "instance"], as_index=False)
            .agg(
                mean_final_best_error=("final_best_error", "mean"),
                std_final_best_error=("final_best_error", "std"),
                mean_total_cost=("total_cost", "mean"),
                std_total_cost=("total_cost", "std"),
                n_seeds=("seed", "nunique"),
            )
        )

    if summary is None or summary.empty:
        return created

    s = summary.copy()
    s["algorithm"] = s["algorithm"].map(normalize_algo_name)

    algo_agg = (
        s.groupby("algorithm", as_index=False)
        .agg(
            mean_final_best_error=("mean_final_best_error", "mean"),
            std_final_best_error=("mean_final_best_error", "std"),
            mean_total_cost=("mean_total_cost", "mean"),
            std_total_cost=("mean_total_cost", "std"),
            n_instances=("instance", "nunique"),
        )
    )
    write_table(algo_agg.sort_values("mean_final_best_error"), out_dirs["tbl"] / "table_lcbench_algorithm_summary")
    created.append("table_lcbench_algorithm_summary")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    sns.boxplot(data=s, x="algorithm", y="mean_final_best_error", hue="algorithm", legend=False, ax=ax, palette="colorblind")
    ax.set_title("LCBench: Final Transformed Objective by Algorithm")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean final transformed objective")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, out_dirs["fig"] / "lcbench_final_error_boxplot_ieee")
    created.append("lcbench_final_error_boxplot_ieee")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    sns.boxplot(data=s, x="algorithm", y="mean_total_cost", hue="algorithm", legend=False, ax=ax, palette="muted")
    ax.set_yscale("log")
    ax.set_title("LCBench: Total Cost by Algorithm")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean total cost (log scale)")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, out_dirs["fig"] / "lcbench_total_cost_boxplot_ieee")
    created.append("lcbench_total_cost_boxplot_ieee")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    for algo, grp in algo_agg.groupby("algorithm"):
        ax.scatter(grp["mean_total_cost"], grp["mean_final_best_error"], s=46, label=algo)
    ax.set_xscale("log")
    ax.set_title("LCBench Pareto View: Cost vs Final Transformed Objective")
    ax.set_xlabel("Mean total cost (log scale)")
    ax.set_ylabel("Mean final transformed objective")
    ax.legend(loc="best")
    save_fig(fig, out_dirs["fig"] / "lcbench_pareto_scatter_ieee")
    created.append("lcbench_pareto_scatter_ieee")

    # Instance-wise winner table and plot
    s_rank = s.copy()
    s_rank["rank"] = s_rank.groupby("instance")["mean_final_best_error"].rank(method="min")
    winners = s_rank[s_rank["rank"] == 1].groupby("algorithm", as_index=False).size()
    winners = winners.rename(columns={"size": "wins"})
    write_table(winners.sort_values("wins", ascending=False), out_dirs["tbl"] / "table_lcbench_instance_wins")
    created.append("table_lcbench_instance_wins")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    sns.barplot(data=winners.sort_values("wins", ascending=False), x="algorithm", y="wins", hue="algorithm", legend=False, ax=ax, palette="viridis")
    ax.set_title("LCBench: Number of Instance Wins")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Count of best-instance wins")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, out_dirs["fig"] / "lcbench_instance_wins_bar_ieee")
    created.append("lcbench_instance_wins_bar_ieee")

    # Runtime from normal per-trial cost (simulated_cost)
    if all_results is not None and not all_results.empty and {"algorithm", "instance", "simulated_cost"}.issubset(all_results.columns):
        ar = all_results[["algorithm", "instance", "simulated_cost"]].copy()
        ar["algorithm"] = ar["algorithm"].map(normalize_algo_name)
        ar["instance"] = pd.to_numeric(ar["instance"], errors="coerce")
        ar["simulated_cost"] = pd.to_numeric(ar["simulated_cost"], errors="coerce")
        ar = ar.dropna(subset=["instance", "simulated_cost"])

        runtime = (
            ar.groupby(["algorithm", "instance"], as_index=False)
            .agg(mean_trial_cost=("simulated_cost", "mean"), std_trial_cost=("simulated_cost", "std"), n=("simulated_cost", "size"))
        )
        runtime["ci95"] = 1.96 * runtime["std_trial_cost"].fillna(0.0) / np.sqrt(runtime["n"].clip(lower=1))

        fig, ax = plt.subplots(figsize=(max(fig_w, 5.2), fig_h), constrained_layout=True)
        for algo, grp in runtime.groupby("algorithm"):
            g = grp.sort_values("instance")
            ax.plot(g["instance"], g["mean_trial_cost"], marker="o", linewidth=1.5, label=algo)
        ax.set_title("LCBench: Normal Cost per Trial by Instance")
        ax.set_xlabel("OpenML task ID (size proxy)")
        ax.set_ylabel("Mean simulated cost per trial")
        ax.legend(loc="best")
        save_fig(fig, out_dirs["fig"] / "lcbench_normal_cost_line_ieee")
        created.append("lcbench_normal_cost_line_ieee")

        write_table(runtime.sort_values(["algorithm", "instance"]), out_dirs["tbl"] / "table_lcbench_normal_cost_by_instance")
        created.append("table_lcbench_normal_cost_by_instance")

    return created


def parse_pathfinding_logs(log_paths: List[Path]) -> pd.DataFrame:
    # Supports both compact and tabular log lines.
    compact = re.compile(r"^\s*([A-Za-z0-9_+-]+)\s*:\s*Cost\s*=\s*([0-9.]+),\s*Valid\s*=\s*([✓xX])\s*,\s*Time\s*=\s*([0-9.]+)s")
    tabular = re.compile(
        r"^\s*([A-Za-z0-9_+-]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(\d+)/(\d+)\s*\|\s*([0-9.]+)s"
    )
    scenario_pat = re.compile(r"Scenario\s*=\s*([A-Z_]+)\s*\|\s*Waypoints\s*=\s*(\d+)")

    rows = []
    for lp in log_paths:
        if not lp.exists():
            continue
        scenario = "unknown"
        waypoints = np.nan
        for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
            m_s = scenario_pat.search(line)
            if m_s:
                scenario = m_s.group(1)
                waypoints = int(m_s.group(2))

            m_c = compact.search(line)
            if m_c:
                algo = normalize_algo_name(m_c.group(1))
                rows.append(
                    {
                        "source_log": lp.name,
                        "scenario": scenario,
                        "waypoints": waypoints,
                        "algorithm": algo,
                        "best_cost": float(m_c.group(2)),
                        "valid_runs": np.nan,
                        "total_runs": np.nan,
                        "runtime_s": float(m_c.group(4)),
                    }
                )
                continue

            m_t = tabular.search(line)
            if m_t:
                algo = normalize_algo_name(m_t.group(1))
                rows.append(
                    {
                        "source_log": lp.name,
                        "scenario": scenario,
                        "waypoints": waypoints,
                        "algorithm": algo,
                        "best_cost": float(m_t.group(5)),
                        "valid_runs": int(m_t.group(6)),
                        "total_runs": int(m_t.group(7)),
                        "runtime_s": float(m_t.group(8)),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["source_log", "scenario", "waypoints", "algorithm", "best_cost", "valid_runs", "total_runs", "runtime_s"])

    df = pd.DataFrame(rows)
    # Keep one record per (log/scenario/waypoints/algorithm): the best-cost smallest runtime pair.
    df = (
        df.sort_values(["source_log", "scenario", "waypoints", "algorithm", "best_cost", "runtime_s"])
        .groupby(["source_log", "scenario", "waypoints", "algorithm"], as_index=False)
        .first()
    )
    return df


def build_pathfinding_figures_tables(root: Path, out_dirs: Dict[str, Path], column: str) -> List[str]:
    created: List[str] = []
    fig_w, fig_h = setup_ieee_style(column)

    logs_dir = root / "results" / "pathfinding" / "raw"
    logs = sorted(logs_dir.glob("*.txt"))
    path_df = parse_pathfinding_logs(logs)

    if path_df.empty:
        return created

    write_table(path_df, out_dirs["tbl"] / "table_pathfinding_parsed_results")
    created.append("table_pathfinding_parsed_results")

    # Aggregate across logs/scenarios
    agg = (
        path_df.groupby("algorithm", as_index=False)
        .agg(
            mean_best_cost=("best_cost", "mean"),
            std_best_cost=("best_cost", "std"),
            mean_runtime_s=("runtime_s", "mean"),
            n_records=("best_cost", "size"),
        )
    )
    write_table(agg.sort_values("mean_best_cost"), out_dirs["tbl"] / "table_pathfinding_algorithm_summary")
    created.append("table_pathfinding_algorithm_summary")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    sns.barplot(data=agg.sort_values("mean_best_cost"), x="algorithm", y="mean_best_cost", hue="algorithm", legend=False, ax=ax, palette="crest")
    ax.set_title("Pathfinding: Mean Best Cost by Algorithm")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean best path cost")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    save_fig(fig, out_dirs["fig"] / "pathfinding_best_cost_bar_ieee")
    created.append("pathfinding_best_cost_bar_ieee")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    sns.barplot(data=agg.sort_values("mean_runtime_s"), x="algorithm", y="mean_runtime_s", hue="algorithm", legend=False, ax=ax, palette="flare")
    ax.set_title("Pathfinding: Mean Runtime by Algorithm")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean runtime (s)")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    save_fig(fig, out_dirs["fig"] / "pathfinding_runtime_bar_ieee")
    created.append("pathfinding_runtime_bar_ieee")

    if path_df["scenario"].notna().any():
        scen = (
            path_df.groupby(["scenario", "algorithm"], as_index=False)
            .agg(mean_best_cost=("best_cost", "mean"), mean_runtime_s=("runtime_s", "mean"))
        )
        fig, ax = plt.subplots(figsize=(max(fig_w, 5.4), fig_h), constrained_layout=True)
        sns.lineplot(data=scen, x="scenario", y="mean_best_cost", hue="algorithm", marker="o", ax=ax)
        ax.set_title("Pathfinding: Scenario-wise Mean Best Cost")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Mean best cost")
        ax.tick_params(axis="x", rotation=25)
        save_fig(fig, out_dirs["fig"] / "pathfinding_scenario_cost_line_ieee")
        created.append("pathfinding_scenario_cost_line_ieee")

    return created


def build_cross_benchmark_tables(root: Path, out_dirs: Dict[str, Path]) -> List[str]:
    created: List[str] = []

    hp_global = safe_read_csv(
        root
        / "results"
        / "hpobench"
        / "statistics_auc"
        / "global_rankings.csv"
    )
    lc_runs = safe_read_csv(root / "results" / "lcbench" / "final_runs.csv")
    lc_summary = None
    if lc_runs is not None and not lc_runs.empty:
        lc_summary = (
            lc_runs.groupby(["algorithm", "instance"], as_index=False)
            .agg(
                mean_final_best_error=("final_best_error", "mean"),
                mean_total_cost=("total_cost", "mean"),
            )
        )

    rows = []
    if hp_global is not None and not hp_global.empty:
        for _, r in hp_global.iterrows():
            rows.append(
                {
                    "benchmark": "HPOBench",
                    "algorithm": normalize_algo_name(r["algorithm"]),
                    "metric_primary": "avg_rank",
                    "value": float(r["avg_rank"]),
                }
            )

    if lc_summary is not None and not lc_summary.empty:
        lc = lc_summary.copy()
        lc["algorithm"] = lc["algorithm"].map(normalize_algo_name)
        agg = lc.groupby("algorithm", as_index=False).agg(
            mean_final_best_error=("mean_final_best_error", "mean"),
            mean_total_cost=("mean_total_cost", "mean"),
        )
        for _, r in agg.iterrows():
            rows.append(
                {
                    "benchmark": "LCBench",
                    "algorithm": r["algorithm"],
                    "metric_primary": "mean_final_best_error",
                    "value": float(r["mean_final_best_error"]),
                }
            )

    if rows:
        cross = pd.DataFrame(rows)
        write_table(cross, out_dirs["tbl"] / "table_cross_benchmark_overview")
        created.append("table_cross_benchmark_overview")

    return created


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate many IEEE-ready Hagfish figures/tables across HPOBench, LCBench, and Pathfinding.")
    repository_root = Path(__file__).resolve().parents[1]
    p.add_argument("--root", type=Path, default=repository_root)
    p.add_argument("--outdir", type=Path, default=repository_root / "results" / "figures")
    p.add_argument("--column", choices=["single", "double"], default="single")
    p.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=list(OUTPUT_FORMATS),
        help="Figure formats to write.",
    )
    return p.parse_args()


def main() -> None:
    global OUTPUT_FORMATS
    args = parse_args()
    OUTPUT_FORMATS = tuple(args.formats)
    out_dirs = ensure_dirs(args.outdir)

    created = []
    created += build_hpobench_figures_tables(args.root, out_dirs, args.column)
    created += build_lcbench_figures_tables(args.root, out_dirs, args.column)
    created += build_pathfinding_figures_tables(args.root, out_dirs, args.column)
    created += build_cross_benchmark_tables(args.root, out_dirs)

    summary_path = args.outdir / "generation_manifest.txt"
    summary_path.write_text("\n".join(created), encoding="utf-8")

    print(f"Generated {len(created)} artifacts")
    print(f"Figures dir: {out_dirs['fig']}")
    print(f"Tables dir: {out_dirs['tbl']}")
    print(f"Manifest: {summary_path}")


if __name__ == "__main__":
    main()
