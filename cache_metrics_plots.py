import json
from typing import List, Literal

from tabulate import tabulate
import glob

import pandas as pd

from load_raw_data import get_raw_metrics, get_cache_metrics_per_sequence

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_correlation_scatter(files, output_dir, filter_mode="all"):
    """
    Generates a separate scatter plot of Cache Hit Rate vs. Log(Execution Time) for each file.
    """
    all_data = {}
    max_time = 0

    # Parse raw data and find the global maximum time for consistent timeout placement
    for path in sorted(files):
        label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")
        # Assuming get_raw_metrics is defined elsewhere in your script
        hit_rates, times, timeouts, requests, results = get_raw_metrics(path, filter_mode=filter_mode)

        if len(times) > 0:
            max_time = max(max_time, np.max(times))

        all_data[label] = (hit_rates, times, timeouts)

    # Calculate timeout y-value (50% higher than the global maximum execution time)
    timeout_y_val = (max_time * 1.5) / 1000 if max_time > 0 else 300

    # 2. Generate and save a separate plot for each file
    for label, (hit_rates, times, timeouts) in all_data.items():
        plt.figure(figsize=(10, 6))

        valid_idx = ~timeouts
        timeout_idx = timeouts

        # Plot standard executions
        plt.scatter(hit_rates[valid_idx], times[valid_idx] / 1000,
                    alpha=0.6, edgecolors='w', linewidth=0.5, label='Completed')

        # Plot timeouts
        if np.any(timeout_idx):
            plt.scatter(hit_rates[timeout_idx], [timeout_y_val] * np.sum(timeout_idx),
                        marker='x', color='red', s=50, label='Timeout')

        # Formatting
        plt.yscale('log')
        plt.xlabel('Cache Hit Rate', fontsize=12)
        plt.ylabel('Execution Time (s, log scale)', fontsize=12)
        plt.title(f'Cache Efficiency vs. Execution Speed: {label}', fontsize=14)
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()

        # Save individual plot
        output_file = f'scatter_cache_vs_time_{label}.png'
        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Scatter plot saved to {output_path}")

def plot_cumulative_churn(files, output_dir, filter_mode="all", ):
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
        output_path = os.path.join(output_dir, f"cumulative_churn_{safe_seq_name}.png")

        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Churn plot saved to {output_path}")


def plot_sequence_cache_state(files, output_dir, filter_mode="all", drop_always_errors=False):
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
        output_path = os.path.join(output_dir, f"cache_state_{safe_seq_name}_{filter_mode}.png")

        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Cache state plot saved to {output_path}")


def calculate_session_hit_rates(files: List[str]) -> pd.DataFrame:
    """
    Parses result files to calculate average hit rates based on session context:
    new session, existing session, or within the current session.
    """
    records = []

    for path in sorted(files):
        algo_label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")

        with open(path, 'r') as file:
            data = json.load(file)

        seen_sessions = set()
        prev_session_id = None

        for entry in data:
            seq_element = entry.get("sequenceElement", {})
            session_info = seq_element.get("session", {})

            if "sessionId" not in session_info:
                continue

            current_session_id = session_info["sessionId"]
            template = seq_element.get("template")

            cache_str = entry.get("@comunica/persistent-cache-manager:sourceState") or \
                        entry.get("@comunica/persistent-cache-manager:sourceStateQuerySource")

            hit_rate = np.nan
            if cache_str:
                try:
                    cache_stats = json.loads(cache_str)
                    hits = cache_stats.get("hits", 0)
                    total = hits + cache_stats.get("misses", 0)
                    hit_rate = (hits / total) if total > 0 else 0.0
                except json.JSONDecodeError:
                    pass

            # Determine session state
            if prev_session_id is None or (
                    current_session_id not in seen_sessions and current_session_id != prev_session_id):
                switch_type = "new_session"
            elif current_session_id == prev_session_id:
                switch_type = "within_session"
            else:
                switch_type = "existing_session"

            if template and not np.isnan(hit_rate):
                records.append({
                    'algorithm': algo_label,
                    'template': template,
                    'switch_type': switch_type,
                    'hit_rate': hit_rate
                })

            seen_sessions.add(current_session_id)
            prev_session_id = current_session_id

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Calculate means and pivot
    avg_df = df.groupby(['algorithm', 'template', 'switch_type'])['hit_rate'].mean().reset_index()

    pivot_df = avg_df.pivot(
        index=['algorithm', 'template'],
        columns='switch_type',
        values='hit_rate'
    ).reset_index()

    pivot_df.columns.name = None

    # Standardize output columns
    expected_columns = ['new_session', 'existing_session', 'within_session']
    for col in expected_columns:
        if col not in pivot_df.columns:
            pivot_df[col] = np.nan

    col_order = ['algorithm', 'template', 'new_session', 'existing_session', 'within_session']
    pivot_df = pivot_df[[c for c in col_order if c in pivot_df.columns]]

    return pivot_df


import os
import json
import pandas as pd
import numpy as np
from typing import List


def calculate_switch_effect(files: List[str]) -> pd.DataFrame:
    """
    Parses result files, normalizes hit rates against template baselines,
    and aggregates the pure effect of session states.
    """
    records = []

    for path in sorted(files):
        algo_label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")

        with open(path, 'r') as file:
            data = json.load(file)

        seen_sessions = set()
        prev_session_id = None

        for entry in data:
            seq_element = entry.get("sequenceElement", {})
            session_info = seq_element.get("session", {})

            if "sessionId" not in session_info:
                continue

            current_session_id = session_info["sessionId"]
            template = seq_element.get("template")

            cache_str = entry.get("@comunica/persistent-cache-manager:sourceState") or \
                        entry.get("@comunica/persistent-cache-manager:sourceStateQuerySource")

            hit_rate = np.nan
            if cache_str:
                try:
                    cache_stats = json.loads(cache_str)
                    hits = cache_stats.get("hits", 0)
                    total = hits + cache_stats.get("misses", 0)
                    hit_rate = (hits / total) if total > 0 else 0.0
                except json.JSONDecodeError:
                    pass

            if prev_session_id is None or (
                    current_session_id not in seen_sessions and current_session_id != prev_session_id):
                switch_type = "new_session"
            elif current_session_id == prev_session_id:
                switch_type = "within_session"
            else:
                switch_type = "existing_session"

            if template and not np.isnan(hit_rate):
                records.append({
                    'algorithm': algo_label,
                    'template': template,
                    'switch_type': switch_type,
                    'hit_rate': hit_rate
                })

            seen_sessions.add(current_session_id)
            prev_session_id = current_session_id

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 1. Calculate the overall baseline hit rate for each algorithm + template combination
    baselines = df.groupby(['algorithm', 'template'])['hit_rate'].mean().reset_index()
    baselines.rename(columns={'hit_rate': 'template_baseline'}, inplace=True)

    # 2. Merge baselines back to the main records
    df = df.merge(baselines, on=['algorithm', 'template'])

    # 3. Calculate the delta (effect of the switch type)
    df['hit_rate_delta'] = df['hit_rate'] - df['template_baseline']

    # 4. Aggregate the deltas by algorithm and switch type across all templates
    effect_df = df.groupby(['algorithm', 'switch_type'])['hit_rate_delta'].mean().reset_index()

    # Pivot for clean presentation
    pivot_df = effect_df.pivot(
        index='algorithm',
        columns='switch_type',
        values='hit_rate_delta'
    ).reset_index()

    pivot_df.columns.name = None

    expected_columns = ['new_session', 'existing_session', 'within_session']
    for col in expected_columns:
        if col not in pivot_df.columns:
            pivot_df[col] = np.nan

    col_order = ['algorithm', 'new_session', 'existing_session', 'within_session']
    pivot_df = pivot_df[[c for c in col_order if c in pivot_df.columns]]

    return pivot_df


import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Literal


def plot_eviction_impact(files: List[str], dep_var: Literal['hit_rate', 'execution_time'],
                         num_bins: int = 100, output_dir=None, file_name=None):
    """
    Parses result files, extracts sequential queries within the same session,
    calculates Pearson correlation, bins the preceding cache eviction percentage,
    and plots the mean delta of the dependent variable for each algorithm.
    """
    records = []

    for path in sorted(files):
        algo_label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")

        with open(path, 'r') as file:
            data = json.load(file)

        for i, entry in enumerate(data):
            seq_element = entry.get("sequenceElement", {})
            session_info = seq_element.get("session", {})
            session_id = session_info.get("sessionId")
            template = seq_element.get("template")

            cache_str = entry.get("@comunica/persistent-cache-manager:sourceState") or \
                        entry.get("@comunica/persistent-cache-manager:sourceStateQuerySource")

            dep_var_value = np.nan
            eviction_pct = np.nan

            if cache_str:
                try:
                    cache_stats = json.loads(cache_str)
                    hits = cache_stats.get("hits", 0)
                    misses = cache_stats.get("misses", 0)
                    total = hits + misses
                    hit_rate = (hits / total) if total > 0 else 0.0

                    eviction_pct = cache_stats.get("evictionPercentage")
                except json.JSONDecodeError:
                    pass
            else:
                hit_rate = np.nan

            if dep_var == 'hit_rate':
                dep_var_value = hit_rate
            elif dep_var == 'execution_time':
                dep_var_value = entry.get("time")
            else:
                raise ValueError(f"Unknown dep_var {dep_var}")

            if session_id is not None and template is not None:
                records.append({
                    'algorithm': algo_label,
                    'session_id': session_id,
                    'template': template,
                    'dep_value': dep_var_value,
                    'eviction_pct': eviction_pct,
                    'query_index': i
                })

    if not records:
        print("No valid data found.")
        return

    df = pd.DataFrame(records)

    # Calculate baselines and delta
    baselines = df.groupby(['algorithm', 'template'])['dep_value'].mean().reset_index()
    baselines.rename(columns={'dep_value': 'baseline_value'}, inplace=True)

    df = df.merge(baselines, on=['algorithm', 'template'])
    df['delta_value'] = df['dep_value'] - df['baseline_value']
    df = df.sort_values(by=['algorithm', 'query_index'])

    # Shift columns to align consecutive queries
    df['prev_session_id'] = df['session_id'].shift(1)
    df['prev_eviction_pct'] = df['eviction_pct'].shift(1)

    within_session_df = df[df['session_id'] == df['prev_session_id']].dropna(
        subset=['prev_eviction_pct', 'delta_value']).copy()

    if within_session_df.empty:
        print("No valid sequential data found.")
        return

    # Calculate Pearson correlation on unbinned data
    correlations = []
    algos = within_session_df['algorithm'].unique()

    for algo in algos:
        subset = within_session_df[within_session_df['algorithm'] == algo]
        r = subset['prev_eviction_pct'].corr(subset['delta_value']) if len(subset) > 1 else np.nan
        correlations.append({'Algorithm': algo, 'Pearson_r': r})

    corr_df = pd.DataFrame(correlations)
    print(f"Correlation Summary (Pearson r) for {dep_var}:")
    print(corr_df.to_string(index=False))

    # Bin the data
    within_session_df['eviction_bin'] = pd.cut(within_session_df['prev_eviction_pct'], bins=num_bins)

    binned_df = within_session_df.groupby(['algorithm', 'eviction_bin'], observed=False).agg(
        mean_delta=('delta_value', 'mean')
    ).reset_index()

    binned_df['bin_midpoint'] = binned_df['eviction_bin'].apply(
        lambda x: x.mid if pd.notnull(x) else np.nan
    )

    # Plot generation
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in algos:
        subset = binned_df[binned_df['algorithm'] == algo].dropna(subset=['mean_delta'])
        algo_r = corr_df.loc[corr_df['Algorithm'] == algo, 'Pearson_r'].values[0]
        label_str = f"{algo} (r={algo_r:.2f})" if pd.notnull(algo_r) else algo

        ax.scatter(
            subset['bin_midpoint'],
            subset['mean_delta'],
            label=label_str,
            marker='o',
            alpha=0.8,
            s=30
        )

    metric_label = "Hit Rate" if dep_var == 'hit_rate' else "Execution Time"

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Preceding Query Cache Evicted (%)', fontsize=12)
    ax.set_ylabel(f'Mean Current Query $\Delta$ {metric_label}', fontsize=12)
    ax.set_title(f'Binned Impact of Preceding Query Evictions on Current {metric_label}', fontsize=14)

    ax.legend(title='Algorithm (Correlation)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_loc = f"eviction_vs_{dep_var}_binned.png"
    if output_dir and file_name:
        output_loc = os.path.join(output_dir, file_name)
    elif output_dir and not file_name:
        output_loc = os.path.join(output_dir, output_loc)
    elif file_name and not output_dir:
        output_loc = file_name

    plt.savefig(output_loc)
    print(f"Plot saved as {output_loc}")


def plot_refinement_sequence_performance(files: List[str], output_dir=None, file_name=None):
    """
    Parses result files to extract refinement sequences (base query + subsequent refinements).
    Calculates and plots the average cumulative hit rate and cumulative evictions per step.
    """
    records = []
    #TODO: This is wrong!
    for path in sorted(files):
        algo_label = os.path.basename(path).replace("query-results-raw-", "").replace(".json", "")

        with open(path, 'r') as file:
            data = json.load(file)

        # Group sequential queries by session
        session_streams = {}
        for entry in data:
            seq_element = entry.get("sequenceElement", {})
            session_id = seq_element.get("session", {}).get("sessionId")

            if session_id is None:
                continue

            ref_metadata = seq_element.get("refinementMetadata") or {}
            is_refinement = bool(ref_metadata.get("patternIds"))

            cache_str = entry.get("@comunica/persistent-cache-manager:sourceState") or \
                        entry.get("@comunica/persistent-cache-manager:sourceStateQuerySource")

            hits, misses, evictions = 0, 0, 0
            if cache_str:
                try:
                    cache_stats = json.loads(cache_str)
                    hits = cache_stats.get("hits", 0)
                    misses = cache_stats.get("misses", 0)
                    evictions = cache_stats.get("evictions", 0)
                except json.JSONDecodeError:
                    pass

            if session_id not in session_streams:
                session_streams[session_id] = []

            session_streams[session_id].append({
                'is_refinement': is_refinement,
                'hits': hits,
                'misses': misses,
                'evictions': evictions
            })

        # Extract sequences and calculate cumulative metrics
        seq_counter = 0
        for session_id, stream in session_streams.items():
            current_seq = []

            for q in stream:
                if not q['is_refinement']:
                    # Base query: evaluate previous sequence and start a new one
                    if len(current_seq) > 1:
                        cum_hits, cum_misses, cum_evictions = 0, 0, 0
                        seq_counter += 1
                        for step, sq in enumerate(current_seq):
                            cum_hits += sq['hits']
                            cum_misses += sq['misses']
                            cum_evictions += sq['evictions']

                            total = cum_hits + cum_misses
                            hr = (cum_hits / total) if total > 0 else np.nan

                            records.append({
                                'algorithm': algo_label,
                                'sequence_id': seq_counter,
                                'step': step,
                                'cum_hit_rate': hr,
                                'cum_evictions': cum_evictions
                            })
                    current_seq = [q]
                else:
                    # Refinement query: append to active sequence
                    if current_seq:
                        current_seq.append(q)

            # Process final sequence in session
            if len(current_seq) > 1:
                cum_hits, cum_misses, cum_evictions = 0, 0, 0
                seq_counter += 1
                for step, sq in enumerate(current_seq):
                    cum_hits += sq['hits']
                    cum_misses += sq['misses']
                    cum_evictions += sq['evictions']

                    total = cum_hits + cum_misses
                    hr = (cum_hits / total) if total > 0 else np.nan

                    records.append({
                        'algorithm': algo_label,
                        'sequence_id': seq_counter,
                        'step': step,
                        'cum_hit_rate': hr,
                        'cum_evictions': cum_evictions
                    })

    if not records:
        print("No valid refinement sequences found.")
        return

    df = pd.DataFrame(records)

    # Calculate means per algorithm per step
    step_stats = df.groupby(['algorithm', 'step']).agg(
        mean_cum_hr=('cum_hit_rate', 'mean'),
        mean_cum_evictions=('cum_evictions', 'mean'),
        n_samples=('sequence_id', 'count')
    ).reset_index()

    # Filter out steps with too few samples to prevent noise in the tails
    min_samples = 5
    step_stats = step_stats[step_stats['n_samples'] >= min_samples]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    algos = step_stats['algorithm'].unique()

    # Ensure 'default' is plotted last so it renders on top
    if 'default' in algos:
        algos = [a for a in algos if a != 'default'] + ['default']

    for algo in algos:
        subset = step_stats[step_stats['algorithm'] == algo].sort_values('step')

        is_default = algo == 'default'
        linewidth = 2.5 if is_default else 1.5
        linestyle = '--' if is_default else '-'
        color = 'black' if is_default else None

        ax1.plot(subset['step'], subset['mean_cum_hr'], label=algo,
                 linewidth=linewidth, linestyle=linestyle, color=color, marker='o', markersize=4)
        ax2.plot(subset['step'], subset['mean_cum_evictions'], label=algo,
                 linewidth=linewidth, linestyle=linestyle, color=color, marker='o', markersize=4)

    # Format Hit Rate Plot
    ax1.set_title('Average Cumulative Hit Rate per Refinement Step')
    ax1.set_xlabel('Refinement Step (0 = Base Query)')
    ax1.set_ylabel('Cumulative Hit Rate')
    ax1.set_xticks(step_stats['step'].unique())
    ax1.grid(True, alpha=0.3)

    # Format Evictions Plot
    ax2.set_title('Average Cumulative Evictions per Refinement Step')
    ax2.set_xlabel('Refinement Step (0 = Base Query)')
    ax2.set_ylabel('Cumulative Evictions')
    ax2.set_xticks(step_stats['step'].unique())
    ax2.grid(True, alpha=0.3)

    # Shared Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title='Algorithm', loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()

    output_loc = "refinement_sequence_performance.png"
    if output_dir and file_name:
        output_loc = os.path.join(output_dir, file_name)
    elif output_dir and not file_name:
        output_loc = os.path.join(output_dir, output_loc)
    elif file_name and not output_dir:
        output_loc = file_name

    plt.savefig(output_loc, bbox_inches='tight')
    print(f"Plot saved as {output_loc}")

def main():
    # Define the path to the benchmark data files
    pattern = os.path.join("data", "query-results-raw-*.json")
    files = glob.glob(pattern)

    if not files:
        print("Error: No data files found matching the pattern.")
        return

    print(f"Found {len(files)} files. Generating plots...")

    # print("Generating correlation scatter plot...")
    # plot_correlation_scatter(files, "output/cache_metric_figures", "no_refinement")
    #
    # print("Generating cumulative churn plots...")
    # plot_cumulative_churn(files, "output/cache_metric_figures", "no_refinement")
    #
    # print("Generating cache state plots...")
    # plot_sequence_cache_state(files, "output/cache_metric_figures")
    # print("All plots generated successfully.")
    #
    # print("Generating hitrate on switch dataframe")
    # df_switches = calculate_session_hit_rates(files)
    # print(tabulate(df_switches, headers='keys', tablefmt='psql'))
    # df_switch_effect = calculate_switch_effect(files)
    # print(tabulate(df_switch_effect, headers='keys', tablefmt='psql'))

    # Plot large cache implementations
    # plot_eviction_impact([file for file in files if "-l" in file],
    #                      dep_var = "hit_rate",
    #                      output_dir="output/cache_metric_figures",
    #                      file_name="eviction_vs_hit_rate_l.png")
    # plot_eviction_impact([file for file in files if "-m" in file],
    #                      dep_var="hit_rate",
    #                      output_dir="output/cache_metric_figures",
    #                      file_name="eviction_vs_hit_rate_m.png")
    # plot_eviction_impact([file for file in files if "-s" in file],
    #                      dep_var="hit_rate",
    #                      output_dir="output/cache_metric_figures",
    #                      file_name="eviction_vs_hit_rate_s.png")

    plot_eviction_impact([file for file in files if "-l" in file],
                         dep_var = "execution_time",
                         num_bins=50,
                         output_dir="output/cache_metric_figures",
                         file_name="eviction_vs_hit_rate_l.png")
    plot_eviction_impact([file for file in files if "-m" in file],
                         dep_var="execution_time",
                         output_dir="output/cache_metric_figures",
                         file_name="eviction_vs_hit_rate_m.png")
    plot_eviction_impact([file for file in files if "-s" in file],
                         dep_var="execution_time",
                         output_dir="output/cache_metric_figures",
                         file_name="eviction_vs_hit_rate_s.png")

    plot_refinement_sequence_performance([file for file in files if "-s" in file],
                         output_dir="output/cache_metric_figures",
                         file_name="refinement_sequence_performance.png")

if __name__ == "__main__":
    main()