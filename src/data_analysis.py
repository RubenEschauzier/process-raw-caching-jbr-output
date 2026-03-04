# Some code for analysis of the data.
import glob
import json
import os
import pandas as pd


def find_failing_queries_per_template(file_path):
    """
    Counts the number of failed queries per template in a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary mapping template names to their failure counts.
    """
    failure_counts = {}

    with open(file_path, 'r') as file:
        data = json.load(file)

    for entry in data:
        if "error" in entry:
            # Safely navigate the nested dictionary
            template = entry.get("sequenceElement", {}).get("template")

            if template:
                # Initialize to 0 if template is new, then add 1
                failure_counts[template] = failure_counts.get(template, 0) + 1

    return dict(sorted(failure_counts.items()))


def compare_failures_across_files(directory_path):
    """
    Reads all .json files in a directory, extracts failure counts,
    and returns a pandas DataFrame for comparison.
    """
    all_results = {}

    # Locate all JSON files in the target directory
    file_pattern = os.path.join(directory_path, '*.json')
    json_files = glob.glob(file_pattern)

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        # Extract failure counts for the current file
        counts = find_failing_queries_per_template(file_path)
        all_results[file_name] = counts

    # Convert the nested dictionary to a DataFrame
    # This automatically aligns the templates (rows) and files (columns)
    df = pd.DataFrame(all_results)

    # Replace NaN values with 0 and convert to integers
    df = df.fillna(0).astype(int)

    # Sort the index (templates) alphabetically
    df = df.sort_index()

    return df

def find_failing_queries(file_path, target_template):
    """
    Extracts failing queries for a specific template from a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        target_template (str): The template name to filter by (e.g., 'interactive-discover-8').

    Returns:
        list: A list of dictionaries representing the failing queries.
    """
    # Load the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)

    failing_queries = []

    # Iterate through entries and filter
    for entry in data:
        if "error" in entry:
            sequence_element = entry.get("sequenceElement", {})
            template = sequence_element.get("template")

            if template == target_template:
                # Remove the clutter
                entry.pop("sequenceInstantiationCounts")
                failing_queries.append(entry)

    return failing_queries

if __name__ == "__main__":
    file_path_default = "data/query-results-raw-default.json"
    file_path_cache_l = "data/query-results-raw-cache-l.json"
    file_path_cache_l_n_b = "data/query-results-raw-cache-n-b-l.json"
    # template_of_interest = "interactive-discover-8"
    # failing_discover_8_cache_l = find_failing_queries(file_path_cache_l_n_b, template_of_interest)
    # failing_discover_8_default = find_failing_queries(file_path_default, template_of_interest)
    failing_queries_all_runs = compare_failures_across_files("data")
    print(failing_queries_all_runs)
