import json
import os
from argparse import ArgumentError
from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import glob
import re

from src.load_raw_data import get_cumulative_data_per_sequence, get_raw_metrics


def plot_cactus(files, output_dir, plotted_value: Literal["exec_time", "http_requests", "results"],
                y_label, title, filter_timeouts = False,
                filter_mode="all", drop_always_errors=True, log_y_axis=True):
    """
    Generates a cactus plot of sorted query execution times.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Generate distinct colors for up to 20 algorithms
    colors = plt.cm.tab20(np.linspace(0, 1, len(files)))

    for idx, path in enumerate(sorted(files)):
        label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")

        # Extract unaggregated metrics
        _, times, timeouts, http_requests, results = get_raw_metrics(
            path,
            filter_mode=filter_mode,
            drop_always_errors=drop_always_errors
        )

        # Filter out timeouts and sort the successful execution times
        if plotted_value == "exec_time":
            if filter_timeouts:
                valid_times = times[~timeouts]
            else:
                valid_times = times
            sorted_values = np.sort(valid_times) / 1000
        elif plotted_value == 'http_requests':
            sorted_values = np.sort(http_requests)
        elif plotted_value == 'results':
            sorted_values = np.sort(results)
        else:
            raise ValueError(f"Invalid argument for plotted_value {plotted_value}")
        plot_data = np.cumsum(sorted_values)
        # X-axis represents the count of successfully solved queries
        x_axis = np.arange(1, len(plot_data) + 1)

        ax.plot(x_axis, plot_data, label=label, color=colors[idx], linewidth=2, alpha=0.9)
    if log_y_axis:
        ax.set_yscale('log')
    ax.set_xlabel('Number of Solved Queries', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Position legend outside to maintain readability for 14 algorithms
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()

    # Format output filename
    output_path = os.path.join(output_dir, f"cactus_plot_{filter_mode}_{plotted_value}.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Cactus plot saved to {output_path}")

def main(output_dir, files=None):
    if not files:
        pattern = os.path.join("data", "query-results-raw-*.json")
        files = glob.glob(pattern)

    if not files:
        print("No data files found.")
        return

    # Structure: {seq_name: {label: {'times': [...], 'results': [...]}}}
    sequence_data = {}

    # 1. Parse files and group data by sequence
    for path in sorted(files):
        label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")
        print(f"Processing {label}...")

        results_per_sequence = get_cumulative_data_per_sequence(
            path, filter_mode="all", drop_always_errors=False
        )

        for seq_name, seq_data in results_per_sequence.items():
            if seq_name not in sequence_data:
                sequence_data[seq_name] = {}

            sequence_data[seq_name][label] = {
                'times': seq_data['cumulative'],
                'results': seq_data['cumulative_results']
            }

    # 2. Generate a dual-axis plot for each sequence
    for seq_name, algorithms in sequence_data.items():
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

        lines = []
        labels = []

        for label, data in algorithms.items():
            # Plot cumulative time on primary y-axis (left)
            line1 = ax1.plot(data['times'] / 1000, label=f"{label} (Time)", linewidth=1.5, alpha=0.8)
            lines.extend(line1)
            labels.append(f"{label} (Time)")

            # Plot cumulative results on secondary y-axis (right)
            line2 = ax2.plot(data['results'], label=f"{label} (Results)", linewidth=1.5, linestyle='--', alpha=0.6)
            lines.extend(line2)
            labels.append(f"{label} (Results)")

        ax1.set_xlabel('Query Index (Step ID)', fontsize=12)
        ax1.set_ylabel('Cumulative Execution Time (s)', fontsize=12)
        ax2.set_ylabel('Cumulative Results', fontsize=12)

        plt.title(f'Cumulative Execution Time and Results: {seq_name}\n(Averaged over Repetitions)', fontsize=14)

        # Combine legends from both axes and place outside the plot
        ax1.legend(lines, labels, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize='small')

        ax1.grid(True, which="both", ls="-", alpha=0.2)
        fig.tight_layout()

        # Format output filename
        safe_seq_name = str(seq_name).replace(" ", "_").lower()
        output_path = os.path.join(output_dir, f"cumulative_plot_{safe_seq_name}.png")

        # Use bbox_inches='tight' to prevent the external legend from being cropped
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    raw_data_default_n_b = os.path.join("data", "query-results-raw-default-n-b.json")
    raw_data_default = os.path.join("data", "query-results-raw-default.json")

    raw_data_cache_l = os.path.join("data", "query-results-raw-cache-l.json")
    raw_data_cache_m = os.path.join("data", "query-results-raw-cache-m.json")
    raw_data_cache_s = os.path.join("data", "query-results-raw-cache-s.json")
    all_locations_cache = [raw_data_default, raw_data_cache_s, raw_data_cache_m, raw_data_cache_l]
    # main(
    #     all_locations_cache
    # )

    raw_data_cache_n_b_l = os.path.join("data", "query-results-raw-cache-n-b-l.json")
    raw_data_cache_n_b_m = os.path.join("data", "query-results-raw-cache-n-b-m.json")
    raw_data_cache_n_b_s = os.path.join("data", "query-results-raw-cache-n-b-s.json")
    all_locations_cache_n_b = [raw_data_default, raw_data_cache_n_b_s, raw_data_cache_n_b_m, raw_data_cache_n_b_l]
    # main(
    #     all_locations_cache_n_b
    # )

    raw_data_query_cache_n_b_l = os.path.join("data", "query-results-raw-query-cache-n-b-l.json")
    raw_data_query_cache_n_b_m = os.path.join("data", "query-results-raw-query-cache-n-b-m.json")
    raw_data_query_cache_n_b_s = os.path.join("data", "query-results-raw-query-cache-n-b-s.json")
    all_locations_query_cache_n_b = [raw_data_default, raw_data_cache_n_b_s, raw_data_cache_n_b_m, raw_data_cache_n_b_l]
    # main(
    #     all_locations_query_cache_n_b
    # )

    raw_data_query_cache_estimate_n_b_l = os.path.join("data", "query-results-raw-query-cache-estimate-n-b-l.json")
    raw_data_query_cache_estimate_n_b_m = os.path.join("data", "query-results-raw-query-cache-estimate-n-b-m.json")
    raw_data_query_cache_estimate_n_b_s = os.path.join("data", "query-results-raw-query-cache-estimate-n-b-s.json")
    all_locations_query_cache_estimate_n_b = [raw_data_default, raw_data_cache_n_b_s, raw_data_cache_n_b_m, raw_data_cache_n_b_l]

    # main(
    #     all_locations_query_cache_estimate_n_b
    # )
    filter_mode = "all"
    plot_cactus(all_locations_cache,
                plotted_value="exec_time",
                y_label="Execution Time (s, log scale)",
                title=f'Cactus Plot: Global Algorithm Performance\nFilter: {filter_mode}',
                filter_timeouts=False,
                filter_mode=filter_mode,
                output_dir="output/execution_time_figures",
                drop_always_errors=True,
                log_y_axis=True)
    plot_cactus(all_locations_cache,
                plotted_value="http_requests",
                y_label="HTTP Requests (s, log scale)",
                title=f'Cactus Plot: Cumulative HTTP requests over all queries\nFilter: {filter_mode}',
                filter_timeouts=False,
                filter_mode=filter_mode,
                output_dir="output/execution_time_figures",
                drop_always_errors=False,
                log_y_axis=True)
    plot_cactus(all_locations_cache,
                plotted_value="results",
                y_label="Produced Results (s, log scale)",
                title=f'Cactus Plot: Cumulative produced results over all queries\nFilter: {filter_mode}',
                filter_timeouts=False,
                filter_mode=filter_mode,
                output_dir="output/execution_time_figures",
                drop_always_errors=False,
                log_y_axis=False)
