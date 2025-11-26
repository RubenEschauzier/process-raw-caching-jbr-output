import os
from functools import partial
import matplotlib.pyplot as plt

from src.load_raw_data import load_json, aggregate_on, average_aggregated_data, average_number, average_list_number, \
    geo_mean_number, geo_mean_list, exclude_non_refinement_pattern, exclude_refinement_pattern, get_geo_means, \
    get_means, execution_time_deviation_from_mean, get_n_errors, get_geo_means_error_filter, has_error_on, get_n_results
from src.visualize_data import plot_algorithm_comparison_v2


def process_raw_data(location):
    data = load_json(location)
    aggregated_on_template = aggregate_on(data, ["sequenceElement", "template"])
    aggregated_on_sequence = aggregate_on(data, ['name'])
    sum_e, prop_e, sum_e_nr, prop_e_nr = get_n_errors(aggregated_on_template)
    a_t, a_ts, a_t_r, a_ts_r, a_t_nr, a_ts_nr = get_means(aggregated_on_template)
    g_t, g_ts, g_t_r, g_ts_r, g_t_nr, g_ts_nr = get_geo_means(aggregated_on_template)
    # g_t_ne, g_ts_ne, g_t_r_ne, g_ts_r_ne, g_t_nr_ne, g_ts_nr_ne = get_geo_means_error_filter(aggregated_on_template)
    n_r, n_rs_r, n_r_nr = get_n_results(aggregated_on_template)
    execution_time_deviation_from_mean(aggregated_on_sequence, a_t)
    return a_t, sum_e, n_r


def main_process_raw_data(locations):
    file_to_time = {}
    file_to_errors = {}
    file_to_results = {}
    for location in locations:
        a_t, sum_e, n_r = process_raw_data(location)
        file_to_time[location] = a_t
        file_to_errors[location] = sum_e
        file_to_results[location] = n_r
    return file_to_time, file_to_errors, file_to_results


def main_process_all_completed(locations):
    aggregated_datas = {}
    errors_in_experiments = []
    for location in locations:
        aggregated_datas[location] = aggregate_on(load_json(location), ["sequenceElement", "template"])
        has_error = has_error_on(aggregated_datas[location])
        errors_in_experiments.append(has_error)
    merged_any_experiment_error = {}
    for key in errors_in_experiments[0]:
        merged_any_experiment_error[key] = [max(values) for values in zip(*(d[key] for d in errors_in_experiments))]

    def exclusion_function_filter_errors(data_point, idx, any_experiment_error):
        template = data_point['sequenceElement']['template']
        if any_experiment_error[template][idx] == 1:
            return True
        return False

    bound_exclusion = partial(exclusion_function_filter_errors, any_experiment_error=merged_any_experiment_error)
    file_to_mean_execution_time = {}
    file_to_errors = {}
    for file, aggregated_data in aggregated_datas.items():
        mean_time_no_error = average_aggregated_data(aggregated_data, "time", average_number, bound_exclusion)

        file_to_mean_execution_time[file] = mean_time_no_error

        sum_e, prop_e, sum_e_nr, prop_e_nr = get_n_errors(aggregated_data)
        file_to_errors[file] = sum_e

    return file_to_mean_execution_time, file_to_errors


if __name__ == "__main__":
    raw_data_location_e_m = os.path.join("data", "query-results-raw-e-m.json")
    raw_data_location_m = os.path.join("data", "query-results-raw-m.json")
    raw_data_location_default = os.path.join("data", "query-results-raw-default.json")
    all_locations = [raw_data_location_default, raw_data_location_m, raw_data_location_e_m]

    mean_exec_time, num_errors, num_results = main_process_raw_data(all_locations)
    fig, ax1, ax2 = plot_algorithm_comparison_v2(mean_exec_time, num_results)
    plt.show()
    mean_exec_times_completed, file_to_errors_output = main_process_all_completed(all_locations)
    fig, ax1, ax2 = plot_algorithm_comparison_v2(mean_exec_times_completed, file_to_errors_output)
    plt.show()