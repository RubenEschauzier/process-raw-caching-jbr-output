import os
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import os
import glob

from load_raw_data import load_json, aggregate_on, get_raw_metrics, get_cache_metrics_per_sequence

import json
from collections import defaultdict


def plot_correlation_scatter(files, filter_mode = "all"):
    """Generates a scatter plot of Cache Hit Rate vs. Log(Execution Time)."""
    plt.figure(figsize=(10, 6))
    all_data = {}
    max_time = 0

    # 1. Parse raw data and find global max time for timeout placement
    for path in sorted(files):
        label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")
        hit_rates, times, timeouts = get_raw_metrics(path, filter_mode=filter_mode)

        if len(times) > 0:
            max_time = max(max_time, np.max(times))
        all_data[label] = (hit_rates, times, timeouts)

    # Place timeouts 50% higher than the maximum recorded execution time
    timeout_y_val = (max_time * 1.5) / 1000 if max_time > 0 else 300

    # 2. Plot data points
    for label, (hit_rates, times, timeouts) in all_data.items():
        valid_idx = ~timeouts
        timeout_idx = timeouts

        # Standard executions
        plt.scatter(hit_rates[valid_idx], times[valid_idx] / 1000,
                    label=label, alpha=0.6, edgecolors='w', linewidth=0.5)

        # Timeouts
        if np.any(timeout_idx):
            plt.scatter(hit_rates[timeout_idx], [timeout_y_val] * np.sum(timeout_idx),
                        marker='x', color='red', s=50, label=f'{label} (Timeout)')

    plt.yscale('log')
    plt.xlabel('Cache Hit Rate', fontsize=12)
    plt.ylabel('Execution Time (s, log scale)', fontsize=12)
    plt.title('Correlation: Cache Efficiency vs. Query Execution Speed', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig('scatter_cache_vs_time.png', dpi=300)
    plt.close()
    print("Scatter plot saved to scatter_cache_vs_time.png")


def plot_cumulative_churn(files, filter_mode="all"):
    """Generates step plots showing cumulative cache evictions per sequence."""
    sequence_data = {}

    # 1. Parse data and calculate cumulative sum of evictions per sequence
    for path in sorted(files):
        label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")
        # Uses the function defined in the previous step
        metrics_per_seq = get_cache_metrics_per_sequence(path, filter_mode=filter_mode)

        for seq_name, metrics in metrics_per_seq.items():
            if seq_name not in sequence_data:
                sequence_data[seq_name] = {}
            sequence_data[seq_name][label] = np.cumsum(metrics['eviction_percentages'])

    # 2. Generate a step plot for each sequence
    for seq_name, algorithms in sequence_data.items():
        plt.figure(figsize=(10, 6))

        for label, cum_evictions in algorithms.items():
            x_axis = np.arange(len(cum_evictions))
            plt.step(x_axis, cum_evictions, where='post', label=label, linewidth=2)

        plt.xlabel('Query Index (Step ID)', fontsize=12)
        plt.ylabel('Cumulative Evictions (%)', fontsize=12)
        plt.title(f'Cumulative Cache Churn: {seq_name}', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()

        safe_seq_name = str(seq_name).replace(" ", "_").lower()
        output_path = f"cumulative_churn_{safe_seq_name}.png"

        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Churn plot saved to {output_path}")


def plot_sequence_cache_state(files, output_path, filter_mode="all", drop_always_errors=False):
    """
    Generates a sequence-aligned plot comparing cache hit rates (lines)
    and eviction percentages (bars).
    """
    sequence_data = {}

    # 1. Parse and group data by sequence
    for path in sorted(files):
        label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")
        metrics_per_seq = get_cache_metrics_per_sequence(
            path, filter_mode=filter_mode, drop_always_errors=drop_always_errors
        )

        for seq_name, metrics in metrics_per_seq.items():
            if seq_name not in sequence_data:
                sequence_data[seq_name] = {}
            sequence_data[seq_name][label] = metrics

    # 2. Generate a plot for each sequence
    for seq_name, algorithms in sequence_data.items():
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        labels = list(algorithms.keys())
        n_algos = len(labels)
        width = 0.8 / n_algos  # Calculate bar width based on number of algorithms

        lines = []
        line_labels = []

        for i, label in enumerate(labels):
            data = algorithms[label]
            # Convert hit rate to percentage to match 0-100 scale
            hitrates = data['hitrates'] * 100
            evictions = data['eviction_percentages']

            x = np.arange(len(hitrates))

            # Offset x coordinates for grouped bars
            offset = (i - n_algos / 2) * width + width / 2

            # Plot evictions as bars on secondary y-axis (right)
            ax2.bar(x + offset, evictions, width=width, alpha=0.3, label=f"{label} (Evictions)")

            # Plot hit rate as lines on primary y-axis (left)
            line, = ax1.plot(x, hitrates, label=f"{label} (Hit Rate)", linewidth=2, marker='o', markersize=4)
            lines.append(line)
            line_labels.append(f"{label} (Hit Rate)")

        # Format axes
        ax1.set_xlabel('Query Index (Step ID)', fontsize=12)
        ax1.set_ylabel('Cache Hit Rate (%)', fontsize=12)
        ax2.set_ylabel('Eviction Percentage (%)', fontsize=12)

        # Enforce 0-100% bounds
        ax1.set_ylim(-5, 105)
        ax2.set_ylim(-5, 105)

        plt.title(f'Sequence-Aligned Cache State: {seq_name}\nFilter: {filter_mode}', fontsize=14)

        # Combine legends
        bars, bar_labels = ax2.get_legend_handles_labels()
        ax1.legend(lines + bars, line_labels + bar_labels, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize='small')

        ax1.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()

        # Format output filename
        safe_seq_name = str(seq_name).replace(" ", "_").lower()
        output_path = os.path.join(output_path, "cache_state_{safe_seq_name}_{filter_mode}.png")

        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Cache state plot saved to {output_path}")



def main():
    # Define the path to the benchmark data files
    pattern = os.path.join("data", "query-results-raw-*.json")
    files = glob.glob(pattern)

    if not files:
        print("Error: No data files found matching the pattern.")
        return

    print(f"Found {len(files)} files. Generating plots...")

    # 1. Generate Correlation Scatter Plot (Global)
    print("Generating correlation scatter plot...")
    plot_correlation_scatter(files, "no_refinement")

    # 2. Generate Cumulative Churn Step Plots (Per Sequence)
    print("Generating cumulative churn plots...")
    plot_cumulative_churn(files, "no_refinement")

    print("Generating cache state plots...")
    plot_sequence_cache_state(files)
    print("All plots generated successfully.")


if __name__ == "__main__":
    main()