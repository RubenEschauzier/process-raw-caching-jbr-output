import json
import re
from collections import defaultdict
from functools import partial
from xxlimited_35 import Null

import numpy as np
from statistics import geometric_mean


def load_json(location):
    with open(location, 'r') as f:
        return json.load(f)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def get_cumulative_data_per_sequence(location, filter_mode="all", drop_always_errors=False):
    """
    Extracts and averages sequence data, including execution time and result counts.

    Parameters:
    location (str): Path to the JSON file.
    filter_mode (str): Filtering strategy. Options: "all", "refinement_only", "no_refinement".
    drop_always_errors (bool): Excludes templates that fail in 100% of their executions.
    """
    data = load_json(location)

    # Pre-calculate template error rates
    always_error_templates = set()
    if drop_always_errors:
        template_stats = defaultdict(lambda: {'total': 0, 'errors': 0})
        for entry in data:
            template = entry.get('sequenceElement', {}).get('template')
            if template:
                template_stats[template]['total'] += 1
                if 'error' in entry:
                    template_stats[template]['errors'] += 1

        always_error_templates = {
            tpl for tpl, stats in template_stats.items()
            if stats['total'] > 0 and stats['total'] == stats['errors']
        }

    aggregated_by_sequence = aggregate_on(data, ['name'])
    results = {}

    for seq_name in sorted(aggregated_by_sequence.keys(), key=natural_sort_key):
        step_times = defaultdict(list)
        step_result_counts = defaultdict(list)

        for entry in aggregated_by_sequence[seq_name]:
            try:
                step_id = int(entry['id'])
                seq_element = entry.get('sequenceElement', {})
                template = seq_element.get('template')

                # Exclude templates with a 100% error rate
                if drop_always_errors and template in always_error_templates:
                    continue

                ref_meta = seq_element.get('refinementMetadata', {})

                # Updated pattern evaluation logic
                has_pattern = len(ref_meta.values()) > 0

                # Apply filtering logic
                if filter_mode == "refinement_only" and not has_pattern:
                    continue
                if filter_mode == "no_refinement" and has_pattern:
                    continue

                step_times[step_id].append(entry.get('time', 0))
                step_result_counts[step_id].append(entry.get('results', 0))

            except (ValueError, KeyError):
                continue

        # Skip sequence if filtering removed all entries
        if not step_times:
            continue

        # Average metrics for each step across repetitions
        sorted_step_ids = sorted(step_times.keys())
        avg_step_times = [np.mean(step_times[sid]) for sid in sorted_step_ids]
        avg_step_results = [np.mean(step_result_counts[sid]) for sid in sorted_step_ids]

        # Store isolated data per sequence
        results[seq_name] = {
            'averages': np.array(avg_step_times),
            'cumulative': np.cumsum(avg_step_times),
            'average_results': np.array(avg_step_results),
            'cumulative_results': np.cumsum(avg_step_results)
        }

    return results

def get_geo_means(aggregated):
    geo_mean_time = average_aggregated_data(aggregated, "time", geo_mean_number, lambda x, i: False)
    geo_mean_timestamps = average_aggregated_data(aggregated, "timestamps", geo_mean_list, lambda x, i: False)
    geo_mean_time_refinement_only = average_aggregated_data(
        aggregated, "time", geo_mean_number, exclude_non_refinement_pattern
    )
    geo_mean_time_timestamps_refinement_only = average_aggregated_data(
        aggregated, "timestamps", geo_mean_list, exclude_non_refinement_pattern
    )
    geo_mean_time_no_refinement = average_aggregated_data(
        aggregated, "time", geo_mean_number, exclude_refinement_pattern
    )
    geo_mean_time_timestamps_no_refinement = average_aggregated_data(
        aggregated, "timestamps", geo_mean_list, exclude_refinement_pattern
    )
    return geo_mean_time, geo_mean_timestamps, geo_mean_time_refinement_only, geo_mean_time_timestamps_refinement_only, \
        geo_mean_time_no_refinement, geo_mean_time_timestamps_no_refinement


def get_geo_means_error_filter(aggregated):
    bound_exclude_non_rf = partial(exclude_non_refinement_pattern, exclude_errors=True)
    bound_exclude_rdf = partial(exclude_refinement_pattern, exclude_errors=True)
    geo_mean_time = average_aggregated_data(aggregated, "time", geo_mean_number, exclude_error_runs)
    geo_mean_timestamps = average_aggregated_data(aggregated, "timestamps", geo_mean_list, exclude_error_runs)
    geo_mean_time_refinement_only = average_aggregated_data(
        aggregated, "time", geo_mean_number, bound_exclude_non_rf
    )
    geo_mean_time_timestamps_refinement_only = average_aggregated_data(
        aggregated, "timestamps", geo_mean_list, bound_exclude_non_rf
    )
    geo_mean_time_no_refinement = average_aggregated_data(
        aggregated, "time", geo_mean_number, bound_exclude_rdf
    )
    geo_mean_time_timestamps_no_refinement = average_aggregated_data(
        aggregated, "timestamps", geo_mean_list, bound_exclude_rdf
    )
    return geo_mean_time, geo_mean_timestamps, geo_mean_time_refinement_only, geo_mean_time_timestamps_refinement_only,\
        geo_mean_time_no_refinement, geo_mean_time_timestamps_no_refinement


def get_means(aggregated):
    average_time = average_aggregated_data(aggregated, "time", average_number, lambda x, i: False)
    average_timestamps = average_aggregated_data(aggregated, "timestamps", average_list_number, lambda x, i: False)

    mean_time_refinement_only = average_aggregated_data(
        aggregated, "time", average_number, exclude_non_refinement_pattern
    )
    mean_time_timestamps_refinement_only = average_aggregated_data(
        aggregated, "timestamps", average_list_number, exclude_non_refinement_pattern
    )

    mean_time_no_refinement = average_aggregated_data(
        aggregated, "time", average_number, exclude_refinement_pattern
    )
    mean_time_timestamps_no_refinement = average_aggregated_data(
        aggregated, "timestamps", average_list_number, exclude_refinement_pattern
    )
    return average_time, average_timestamps, mean_time_refinement_only, mean_time_timestamps_refinement_only, \
        mean_time_no_refinement, mean_time_timestamps_no_refinement


def get_n_errors(aggregated):
    def sum_errors(aggregated_to_average):
        errors_to_number = [1 if ele is not None else 0 for ele in aggregated_to_average]
        return sum(errors_to_number)

    def average_proportion_errors(aggregated_to_average):
        errors_to_number = [1 if ele is not None else 0 for ele in aggregated_to_average]
        return sum(errors_to_number) / len(errors_to_number)
    error_messages = {
        key: [
            {
                "error_msg": execution_output["error"] if "error" in execution_output else None,
                "time": execution_output["time"],
                "results": execution_output["results"]
            }
            for execution_output in value
        ]
        for key, value in aggregated.items()
    }
    summed_errors = average_aggregated_data(aggregated, "error", sum_errors, lambda x, i: False)
    proportion_errors = average_aggregated_data(aggregated, "error", average_proportion_errors, lambda x, i: False)

    sum_errors_no_refinement = average_aggregated_data(aggregated, "error",
                                                       sum_errors,
                                                       exclude_refinement_pattern)
    proportion_errors_no_refinement = average_aggregated_data(aggregated, "error",
                                                              average_proportion_errors,
                                                              exclude_refinement_pattern)
    return summed_errors, proportion_errors, sum_errors_no_refinement, proportion_errors_no_refinement


def get_cache_metrics_per_sequence(location, filter_mode="all", drop_always_errors=False, timeout_ms=180000):
    data = load_json(location)

    # Pre-calculate template error rates, including timeouts
    always_error_templates = set()
    if drop_always_errors:
        template_stats = defaultdict(lambda: {'total': 0, 'errors': 0})
        for entry in data:
            template = entry.get('sequenceElement', {}).get('template')
            if template:
                template_stats[template]['total'] += 1
                if 'error' in entry or entry.get('time', 0) >= timeout_ms:
                    template_stats[template]['errors'] += 1

        always_error_templates = {
            tpl for tpl, stats in template_stats.items()
            if stats['total'] > 0 and stats['total'] == stats['errors']
        }

    aggregated_by_sequence = aggregate_on(data, ['name'])
    results = {}

    for seq_name in sorted(aggregated_by_sequence.keys(), key=natural_sort_key):
        step_hitrates = defaultdict(list)
        step_evictions = defaultdict(list)

        for entry in aggregated_by_sequence[seq_name]:
            try:
                step_id = int(entry['id'])
                seq_element = entry.get('sequenceElement', {})
                template = seq_element.get('template')

                # Apply error filtering
                if drop_always_errors and template in always_error_templates:
                    continue

                ref_meta = seq_element.get('refinementMetadata', {})
                has_pattern = len(ref_meta.values()) > 0

                # Apply mode filtering
                if filter_mode == "refinement_only" and not has_pattern:
                    continue
                if filter_mode == "no_refinement" and has_pattern:
                    continue

                # Parse cache states
                cache_state_raw = entry.get('@comunica/persistent-cache-manager:sourceState', '{}')
                cache_state = json.loads(cache_state_raw)

                hits = cache_state.get('hits', 0)
                misses = cache_state.get('misses', 0)
                eviction_pct = cache_state.get('evictionPercentage', 0)
                http_requests = entry.get('httpRequests', 0)

                denominator = misses + http_requests
                hitrate = hits / denominator if denominator > 0 else 0.0

                step_hitrates[step_id].append(hitrate)
                step_evictions[step_id].append(eviction_pct)

            except (ValueError, KeyError, json.JSONDecodeError):
                continue

        # Skip empty sequences
        if not step_hitrates:
            continue

        sorted_step_ids = sorted(step_hitrates.keys())
        avg_hitrates = [np.mean(step_hitrates[sid]) for sid in sorted_step_ids]
        avg_evictions = [np.mean(step_evictions[sid]) for sid in sorted_step_ids]

        results[seq_name] = {
            'hitrates': np.array(avg_hitrates),
            'eviction_percentages': np.array(avg_evictions)
        }

    return results


def get_raw_metrics(location, filter_mode="all", drop_always_errors=False, timeout_ms=180000):
    data = load_json(location)

    # Pre-calculate template error rates, including timeouts
    always_error_templates = set()
    if drop_always_errors:
        template_stats = defaultdict(lambda: {'total': 0, 'errors': 0})
        for entry in data:
            template = entry.get('sequenceElement', {}).get('template')
            if template:
                template_stats[template]['total'] += 1
                if 'error' in entry or entry.get('time', 0) >= timeout_ms:
                    template_stats[template]['errors'] += 1

        always_error_templates = {
            tpl for tpl, stats in template_stats.items()
            if stats['total'] > 0 and stats['total'] == stats['errors']
        }

    hit_rates, times, timeouts = [], [], []

    for entry in data:
        try:
            seq_element = entry.get('sequenceElement', {})
            template = seq_element.get('template')

            # Apply error filtering
            if drop_always_errors and template in always_error_templates:
                continue

            ref_meta = seq_element.get('refinementMetadata', {})
            has_pattern = len(ref_meta.values()) > 0

            # Apply mode filtering
            if filter_mode == "refinement_only" and not has_pattern:
                continue
            if filter_mode == "no_refinement" and has_pattern:
                continue

            time_ms = entry.get('time', 0)
            is_timeout = 'error' in entry or time_ms >= timeout_ms

            cache_state_raw = entry.get('@comunica/persistent-cache-manager:sourceState', '{}')
            cache_state = json.loads(cache_state_raw)

            hits = cache_state.get('hits', 0)
            misses = cache_state.get('misses', 0)
            http_requests = entry.get('httpRequests', 0)

            denominator = misses + http_requests
            hitrate = hits / denominator if denominator > 0 else 0.0

            hit_rates.append(hitrate)
            times.append(time_ms)
            timeouts.append(is_timeout)

        except (ValueError, KeyError, json.JSONDecodeError):
            continue

    return np.array(hit_rates), np.array(times), np.array(timeouts)


def get_n_results(aggregated):
    mean_results = average_aggregated_data(aggregated, "results", average_number, lambda x, i: False)
    mean_results_rf_only = average_aggregated_data(
        aggregated, "results", average_number, exclude_non_refinement_pattern
    )
    mean_results_no_refinement = average_aggregated_data(
        aggregated, "results", average_number, exclude_refinement_pattern
    )
    return mean_results, mean_results_rf_only, mean_results_no_refinement


def execution_time_deviation_from_mean(aggregated_per_sequence, template_to_mean):
    max_len_sequence = max([len(seq) for seq in aggregated_per_sequence.values()])
    deviation_at_location = [[] for i in range(max_len_sequence)]
    for sequence, elements in aggregated_per_sequence.items():
        for i, element in enumerate(elements):
            template = element['sequenceElement']['template']
            mean_exec_time = template_to_mean[template]
            element_exec_time = element['time']
            deviation = element_exec_time / mean_exec_time
            deviation_at_location[i].append(deviation)
    mean_deviation_at_location = [np.mean(at_location) for at_location in deviation_at_location]
    return deviation_at_location, mean_deviation_at_location


def errors_deviation_from_mean(aggregated_per_sequence, template_to_errors_probability):
    pass


def aggregate_on(data, aggregate_keys, selection_keys=None):
    aggregated = {}
    for data_point in data:
        aggregation_value = find_at_keys(data_point, aggregate_keys)
        if not hashable(aggregation_value):
            raise ValueError("Value associated with aggregation keys not hashable")

        if aggregation_value not in aggregated:
            aggregated[aggregation_value] = []
        if selection_keys:
            to_save = find_at_keys(data_point, selection_keys)
        else:
            to_save = data_point
        aggregated[aggregation_value].append(to_save)
    return aggregated


def average_aggregated_data(aggregated, average_key, average_function, exclusion_function):
    averaged_results = {}
    for agg_key, value in aggregated.items():
        selected = []
        for i, data in enumerate(value):
            if not exclusion_function(data, i):
                if average_key in data:
                    selected.append(data[average_key])
                else:
                    selected.append(None)
        average = average_function(selected)
        averaged_results[agg_key] = average
    return averaged_results

# TODO: Within sequence look into deviation from average over sequence elements, to show cache working or not (done)
# TODO: Deviation of mean within refinements sequence (done)
# TODO: Average with errors filtered
# TODO: Average/Geo time outside of refinements (done)
# TODO: Average/Geo time total (done)
def has_error_on(aggregated_on_template):
    has_error = {}
    for template, values in aggregated_on_template.items():
        error_in_execution = [1 if 'error' in value else 0 for value in values]
        has_error[template] = error_in_execution
    return has_error


def average_number(aggregated_to_average):
    if len(aggregated_to_average) == 0:
        return -1
    return np.mean(aggregated_to_average)


def geo_mean_number(aggregated_to_average):
    if len(aggregated_to_average) == 0:
        return -1
    return geometric_mean(aggregated_to_average)


def geo_mean_list(aggregated_to_average):
    if len(aggregated_to_average) == 0:
        return [-1]
    max_len = max([len(sub_list) for sub_list in aggregated_to_average])
    grouped_timestamps = []
    for i in range(max_len):

        geo_mean_ts_i = []
        for instantiation_result in aggregated_to_average:
            if len(instantiation_result) > i:
                geo_mean_ts_i.append(instantiation_result[i])

        grouped_timestamps.append(geometric_mean(geo_mean_ts_i))
    return grouped_timestamps


def average_list_number(aggregated_to_average):
    if len(aggregated_to_average) == 0:
        return [-1]
    max_len = max([len(sub_list) for sub_list in aggregated_to_average])
    grouped_timestamps = []
    for i in range(max_len):

        geo_mean_ts_i = []
        for instantiation_result in aggregated_to_average:
            if len(instantiation_result) > i:
                geo_mean_ts_i.append(instantiation_result[i])

        grouped_timestamps.append(np.mean(geo_mean_ts_i))
    return grouped_timestamps


def exclude_non_refinement_pattern(data_point, idx, exclude_errors=False):
    if exclude_errors and "error" in data_point:
        return True
    refinement_patterns = data_point['sequenceElement']['refinementMetadata']
    return len(refinement_patterns.keys()) == 0


def exclude_refinement_pattern(data_point, idx, exclude_errors=False):
    if exclude_errors and "error" in data_point:
        return True
    return not exclude_non_refinement_pattern(data_point, idx, False)


def exclude_error_runs(data_point, idx):
    if "error" in data_point:
        return True
    return False


def find_at_keys(data, keys):
    for key in keys:
        data = data[key]
    return data


def hashable(v):
    try:
        hash(v)
    except TypeError:
        return False
    return True
