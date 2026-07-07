import os
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_md5_hash(query_string: str) -> str:
    return hashlib.md5(query_string.encode('utf-8')).hexdigest()


def simulate_lru_cache(sequence_queries_nodes, cache_size):
    if not sequence_queries_nodes:
        return 0.0

    hits = 0
    total_accesses = 0
    cache = []

    for query_nodes in sequence_queries_nodes:
        for node in query_nodes:
            total_accesses += 1
            if node in cache:
                hits += 1
                cache.remove(node)
                cache.append(node)
            else:
                cache.append(node)
                if len(cache) > cache_size:
                    cache.pop(0)

    return hits / total_accesses if total_accesses > 0 else 0.0


def compute_jaccard_stats(sequence_queries_nodes):
    if len(sequence_queries_nodes) < 2:
        return 0.0, 0.0

    jaccards = []
    for i in range(1, len(sequence_queries_nodes)):
        q_prev = sequence_queries_nodes[i - 1]
        q_curr = sequence_queries_nodes[i]
        union_size = len(q_prev.union(q_curr))
        if union_size > 0:
            jaccards.append(len(q_prev.intersection(q_curr)) / union_size)
        else:
            jaccards.append(0.0)

    if jaccards:
        return np.mean(jaccards), np.std(jaccards)
    return 0.0, 0.0


def analyze_sweep():
    sweep_dir = Path("/home/ruben-eschauzier/projects/process-caching-journal/data/sweep-results")
    runs = sorted(list(sweep_dir.glob("run_*")))

    rows = []

    for run in runs:
        run_name = run.name

        metadata_file = run / "sweep_metadata.json"
        if not metadata_file.exists():
            continue

        with open(metadata_file, "r") as f:
            meta = json.load(f)

        hparams = meta.get("hyperparameters", {})

        mean_len = hparams.get("sequenceGenerator.meanLogSequenceLength", 3)
        std_len = hparams.get("sequenceGenerator.stdLogSequenceLength", 0.2)
        mean_trans = hparams.get("sequenceGenerator.meanLogTransitionProbability", -2)

        sparql_dir = run / "generated" / "out-queries"
        topology_dir = run / "combinations" / "combination_0" / "output-topology-tracking"

        if not sparql_dir.exists() or not topology_dir.exists():
            print(f"Skipping {run_name} due to missing directories")
            continue

        sparql_files = sorted(list(sparql_dir.glob("*.sparql")))

        for sparql_file in sparql_files:
            sequence_name = sparql_file.stem

            with open(sparql_file, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                queries = [q.strip() for q in content.split('\n\n') if q.strip()]

            seq_queries_nodes = []
            for query in queries:
                q_hash = generate_md5_hash(query)
                matching_files = [
                    f for f in topology_dir.glob(f"*-{q_hash}.json")
                    if not f.name.endswith('.tmp')
                ]
                matching_files.sort(key=lambda p: int(p.name.split('-')[0]))

                visited_in_query = set()
                for topo_file in matching_files:
                    try:
                        with open(topo_file, 'r', encoding='utf-8') as tf:
                            topo = json.load(tf)
                            nodes = set(topo.get("indexToNodeDict", {}).values())
                            nodes.discard("root")
                            visited_in_query.update(nodes)
                    except Exception:
                        pass
                seq_queries_nodes.append(visited_in_query)

            seq_length = len(queries)
            all_visited = set().union(*seq_queries_nodes) if seq_queries_nodes else set()
            total_unique_sources = len(all_visited)

            jaccard_mean, jaccard_std = compute_jaccard_stats(seq_queries_nodes)

            new_sources_list = []
            visited_so_far = set()
            for q_nodes in seq_queries_nodes:
                new_sources_list.append(len(q_nodes - visited_so_far))
                visited_so_far.update(q_nodes)

            avg_new_sources = np.mean(new_sources_list) if new_sources_list else 0.0
            std_new_sources = np.std(new_sources_list) if new_sources_list else 0.0

            # Dynamic threshold: Mean + 2 Standard Deviations
            dynamic_threshold = avg_new_sources + (2 * std_new_sources)
            large_increases_count = sum(1 for x in new_sources_list if x > dynamic_threshold and x > 0)

            rows.append({
                "Run": run_name,
                "Sequence": sequence_name,
                "meanLogSequenceLength": mean_len,
                "stdLogSequenceLength": std_len,
                "meanLogTransitionProbability": mean_trans,
                "Sequence Length": seq_length,
                "Total Unique Sources": total_unique_sources,
                "Avg New Sources": avg_new_sources,
                "Large Increases Count": large_increases_count,
                "Jaccard Overlap Mean": jaccard_mean,
                "Jaccard Overlap Std": jaccard_std
            })

    df = pd.DataFrame(rows)
    return df


def generate_sweep_plots(df):
    plt.style.use("seaborn-v0_8-whitegrid")

    hparams = [
        ("meanLogSequenceLength", "Mean Log Sequence Length"),
        ("stdLogSequenceLength", "Std Log Sequence Length"),
        ("meanLogTransitionProbability", "Mean Log Transition Prob")
    ]

    metrics = [
        ("Jaccard Overlap Mean", "Avg Jaccard Overlap", "#4C72B0"),
        ("Jaccard Overlap Std", "Std Jaccard Overlap", "#DD8452"),
        ("Avg New Sources", "Avg New Sources / Query", "#55A868"),
        ("Large Increases Count", "Count of Large Source Jumps", "#C44E52")
    ]

    fig, axes = plt.subplots(len(hparams), len(metrics), figsize=(18, 12))

    for i, (hp_col, hp_label) in enumerate(hparams):
        if hp_col == "meanLogSequenceLength":
            df_filtered = df[(df["stdLogSequenceLength"] == 0.2) & (df["meanLogTransitionProbability"] == -2)]
        elif hp_col == "stdLogSequenceLength":
            df_filtered = df[(df["meanLogSequenceLength"] == 3) & (df["meanLogTransitionProbability"] == -2)]
        elif hp_col == "meanLogTransitionProbability":
            df_filtered = df[(df["meanLogSequenceLength"] == 3) & (df["stdLogSequenceLength"] == 0.2)]

        for j, (metric_col, metric_label, color) in enumerate(metrics):
            ax = axes[i, j]

            grouped = df_filtered.groupby(hp_col)[metric_col].agg(["mean", "std"]).reset_index()
            grouped["std"] = grouped["std"].fillna(0)

            ax.errorbar(
                grouped[hp_col],
                grouped["mean"],
                yerr=grouped["std"],
                fmt='-o',
                color=color,
                linewidth=2.5,
                markersize=8,
                capsize=5,
                elinewidth=2
            )

            ax.set_title(f"{metric_label} vs {hp_col}", fontsize=12, fontweight="bold")
            ax.set_xlabel(hp_label, fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)

            unique_vals = sorted(df_filtered[hp_col].unique())
            ax.set_xticks(unique_vals)

    plt.suptitle("Hyperparameter Sweep Analysis on Sequence Topology Metrics", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()

    output_dir = Path("/home/ruben-eschauzier/projects/process-caching-journal/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hyperparameter_sweep_analysis.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    df = analyze_sweep()
    generate_sweep_plots(df)

    summary_path = Path(
        "/home/ruben-eschauzier/projects/process-caching-journal/output/hyperparameter_sweep_summary.csv")
    df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()