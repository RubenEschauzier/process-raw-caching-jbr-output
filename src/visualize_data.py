import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def plot_algorithm_comparison(algo_times, algo_errors, figsize=(16, 8)):
    """
    Create a grouped bar chart comparing algorithm execution times across queries.

    Parameters:
    -----------
    algo_times : dict
        Dictionary mapping algorithm names to query execution times
    algo_errors : dict
        Dictionary mapping algorithm names to query error counts
    figsize : tuple
        Figure size (width, height)
    """

    # Get all queries and sort them by category
    all_queries = sorted(list(algo_times[list(algo_times.keys())[0]].keys()),
                         key=lambda x: (x.split('-')[1], int(x.split('-')[2])))

    algorithms = list(algo_times.keys())
    n_algos = len(algorithms)
    n_queries = len(all_queries)

    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Bar settings
    bar_width = 0.8 / n_algos
    x = np.arange(n_queries)

    # Color palette - use colorblind-friendly colors
    colors = plt.cm.Set2(np.linspace(0, 1, n_algos))

    # Minimum value for log scale (for -1 timeouts)
    min_valid_time = min([t for algo_dict in algo_times.values()
                          for t in algo_dict.values() if t > 0])
    timeout_value = min_valid_time * 0.5  # Plot timeouts below minimum

    # Plot bars for each algorithm
    for i, algo in enumerate(algorithms):
        times = []
        has_errors = []

        for query in all_queries:
            time = algo_times[algo][query]
            error = algo_errors[algo][query]

            # Handle timeouts (-1)
            if time == -1:
                times.append(timeout_value)
            else:
                times.append(time)

            has_errors.append(error > 0)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, times, bar_width, label=algo,
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add hatching for queries with errors
        for j, (bar, has_error) in enumerate(zip(bars, has_errors)):
            if has_error:
                bar.set_hatch('///')

            # Mark timeouts with red 'X'
            if algo_times[algo][all_queries[j]] == -1:
                bar.set_facecolor('none')
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
                ax.text(bar.get_x() + bar.get_width() / 2, timeout_value * 1.5,
                        'X', ha='center', va='bottom', color='red',
                        fontweight='bold', fontsize=10)

    # Formatting
    ax.set_xlabel('Query Template', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Performance Comparison Across Query Templates',
                 fontsize=14, fontweight='bold', pad=20)

    # # Set log scale
    # ax.set_yscale('log')

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace('interactive-', '') for q in all_queries],
                       rotation=45, ha='right')

    # Add vertical line to separate query types
    discover_idx = next(i for i, q in enumerate(all_queries) if 'discover' in q)
    ax.axvline(x=discover_idx - 0.5, color='gray', linestyle='--',
               linewidth=1.5, alpha=0.5)

    # Add text labels for query types
    ax.text(discover_idx / 2 - 0.5, ax.get_ylim()[1] * 0.95, 'Short Queries',
            ha='center', fontsize=11, style='italic', alpha=0.7)
    ax.text((discover_idx + n_queries) / 2 - 0.5, ax.get_ylim()[1] * 0.95,
            'Discover Queries', ha='center', fontsize=11, style='italic', alpha=0.7)

    # Legend with custom patches
    legend_elements = [mpatches.Patch(color=colors[i], label=algo, alpha=0.8)
                       for i, algo in enumerate(algorithms)]
    legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='black',
                                          hatch='///', label='Has Errors'))
    legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='red',
                                          linewidth=2, label='Timeout'))

    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9,
              fontsize=10)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig, ax


def plot_algorithm_comparison_v2(algo_times, algo_errors, figsize=(16, 8)):
    """
    Create a grouped bar chart comparing algorithm execution times across queries.
    Includes a secondary y-axis showing error counts as markers.

    Parameters:
    -----------
    algo_times : dict
        Dictionary mapping algorithm names to query execution times
    algo_errors : dict
        Dictionary mapping algorithm names to query error counts
    figsize : tuple
        Figure size (width, height)
    """

    # Get all queries and sort them by category
    all_queries = sorted(list(algo_times[list(algo_times.keys())[0]].keys()),
                         key=lambda x: (x.split('-')[1], int(x.split('-')[2])))

    algorithms = list(algo_times.keys())
    n_algos = len(algorithms)
    n_queries = len(all_queries)

    # Set up the plot with twin axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()  # Create second y-axis

    # Bar settings
    bar_width = 0.8 / n_algos
    x = np.arange(n_queries)

    # Color palette - use colorblind-friendly colors
    colors = plt.cm.Set2(np.linspace(0, 1, n_algos))

    # Minimum value for log scale (for -1 timeouts)
    min_valid_time = min([t for algo_dict in algo_times.values()
                          for t in algo_dict.values() if t > 0])
    timeout_value = min_valid_time * 0.5  # Plot timeouts below minimum

    # Plot bars for each algorithm (execution time)
    for i, algo in enumerate(algorithms):
        times = []
        errors = []

        for query in all_queries:
            time = algo_times[algo][query]
            error = algo_errors[algo][query]

            # Handle timeouts (-1)
            if time == -1:
                times.append(timeout_value)
            else:
                times.append(time)

            errors.append(error)

        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax1.bar(x + offset, times, bar_width, label=algo,
                       color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)

        # Plot error counts as text labels on secondary axis
        error_x = x + offset
        error_y = np.array(errors)

        # Only show labels where errors > 0
        for j, (ex, ey) in enumerate(zip(error_x, error_y)):
            if ey > 0:
                ax2.text(ex, ey, str(int(ey)), ha='center', va='center',
                         fontsize=9, fontweight='bold', color=colors[i],
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                   edgecolor=colors[i], linewidth=2, alpha=0.9),
                         zorder=10)

        # Mark timeouts with red 'X' on bars
        for j, bar in enumerate(bars):
            if algo_times[algo][all_queries[j]] == -1:
                bar.set_facecolor('none')
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
                ax1.text(bar.get_x() + bar.get_width() / 2, timeout_value * 1.5,
                         'X', ha='center', va='bottom', color='red',
                         fontweight='bold', fontsize=10)

    # Formatting for primary y-axis (execution time)
    ax1.set_xlabel('Query Template', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Algorithm Performance Comparison: Execution Time and Error Counts',
                  fontsize=14, fontweight='bold', pad=20)

    # # Set log scale for execution time
    # ax1.set_yscale('log')

    # Formatting for secondary y-axis (error counts)
    ax2.set_ylabel('Number of Errors', fontsize=12, fontweight='bold', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')

    # Set appropriate range for error axis
    max_errors = max([max(algo_errors[algo].values()) for algo in algorithms])
    ax2.set_ylim(-0.5, max_errors + 1)

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels([q.replace('interactive-', '') for q in all_queries],
                        rotation=45, ha='right')

    # Add vertical line to separate query types
    discover_idx = next(i for i, q in enumerate(all_queries) if 'discover' in q)
    ax1.axvline(x=discover_idx - 0.5, color='gray', linestyle='--',
                linewidth=1.5, alpha=0.5)

    # Add text labels for query types
    y_pos = ax1.get_ylim()[1] * 0.95
    ax1.text(discover_idx / 2 - 0.5, y_pos, 'Short Queries',
             ha='center', fontsize=11, style='italic', alpha=0.7)
    ax1.text((discover_idx + n_queries) / 2 - 0.5, y_pos,
             'Discover Queries', ha='center', fontsize=11, style='italic', alpha=0.7)

    # Legend
    legend_elements = [mpatches.Patch(color=colors[i], label=algo, alpha=0.7)
                       for i, algo in enumerate(algorithms)]
    legend_elements.append(mpatches.Patch(facecolor='none', edgecolor='red',
                                          linewidth=2, label='Timeout'))

    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.95,
               fontsize=10)

    # Add note about error counts
    ax1.text(0.02, 0.98, 'Numbers show error counts (right axis)',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Grid (only on primary axis)
    ax1.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.set_axisbelow(True)

    plt.tight_layout()
    return fig, ax1, ax2

# Alternative: Create a heatmap version
def plot_heatmap_comparison(algo_times, figsize=(12, 8)):
    """Create a heatmap showing relative performance."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    algorithms = list(algo_times.keys())
    queries = sorted(list(algo_times[algorithms[0]].keys()))

    # Create matrix
    data = np.zeros((len(algorithms), len(queries)))
    for i, algo in enumerate(algorithms):
        for j, query in enumerate(queries):
            time = algo_times[algo][query]
            data[i, j] = time if time > 0 else np.nan

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd',
                   norm=LogNorm(vmin=np.nanmin(data), vmax=np.nanmax(data)))

    # Labels
    ax.set_xticks(np.arange(len(queries)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels([q.replace('interactive-', '') for q in queries],
                       rotation=45, ha='right')
    ax.set_yticklabels(algorithms)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Execution Time (ms)', rotation=270, labelpad=20)

    ax.set_title('Algorithm Performance Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Example data structure - replace with your actual data
    algo_times = {
        'Algorithm_A': {'interactive-short-3': 3003.82, 'interactive-short-1': 2682.36,
                        'interactive-discover-8': 11720.48, 'interactive-short-4': 176.85,
                        'interactive-short-6': 38785.0, 'interactive-discover-7': 807.42,
                        'interactive-discover-1': 1372.72, 'interactive-discover-4': 1362.76,
                        'interactive-discover-6': 4166.36, 'interactive-discover-2': 1489.18,
                        'interactive-short-7': -1, 'interactive-short-5': 1455.79,
                        'interactive-discover-5': 1825.28, 'interactive-discover-3': 6768.08,
                        'interactive-short-2': 508.33},
        # Add your other 8 algorithms here
        'Algorithm_B': {'interactive-short-3': 2500.0, 'interactive-short-1': 2200.0,
                        'interactive-discover-8': 10000.0, 'interactive-short-4': 200.0,
                        'interactive-short-6': 35000.0, 'interactive-discover-7': 900.0,
                        'interactive-discover-1': 1500.0, 'interactive-discover-4': 1400.0,
                        'interactive-discover-6': 4000.0, 'interactive-discover-2': 1600.0,
                        'interactive-short-7': 1200.0, 'interactive-short-5': 1300.0,
                        'interactive-discover-5': 2000.0, 'interactive-discover-3': 7000.0,
                        'interactive-short-2': 600.0},
    }

    algo_errors = {
        'Algorithm_A': {'interactive-short-3': 0, 'interactive-short-1': 2,
                        'interactive-discover-8': 1, 'interactive-short-4': 0,
                        'interactive-short-6': 0, 'interactive-discover-7': 0,
                        'interactive-discover-1': 0, 'interactive-discover-4': 1,
                        'interactive-discover-6': 0, 'interactive-discover-2': 0,
                        'interactive-short-7': 5, 'interactive-short-5': 0,
                        'interactive-discover-5': 0, 'interactive-discover-3': 0,
                        'interactive-short-2': 0},
        'Algorithm_B': {'interactive-short-3': 1, 'interactive-short-1': 0,
                        'interactive-discover-8': 0, 'interactive-short-4': 0,
                        'interactive-short-6': 2, 'interactive-discover-7': 0,
                        'interactive-discover-1': 0, 'interactive-discover-4': 0,
                        'interactive-discover-6': 1, 'interactive-discover-2': 0,
                        'interactive-short-7': 0, 'interactive-short-5': 0,
                        'interactive-discover-5': 0, 'interactive-discover-3': 1,
                        'interactive-short-2': 0},
    }

    # Create the plot
    fig, ax1, ax2 = plot_algorithm_comparison_v2(algo_times, algo_errors)
    # plt.savefig('algorithm_comparison.pdf', dpi=300, bbox_inches='tight')
    # plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig_heat, ax_heat = plot_heatmap_comparison(algo_times)
    plt.show()