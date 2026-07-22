#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np

def parse_log_line(line, line_num, filepath):
    line = line.strip()
    if not line:
        return None
    try:
        # Define a safe evaluation namespace that handles numpy types and special constants
        ns = {
            'np': np,
            'nan': np.nan,
            'float': float,
            'int': int,
            'pd': pd,
        }
        # Use eval to evaluate the string representation of the dictionary
        parsed = eval(line, ns)
        if isinstance(parsed, dict):
            return parsed
        else:
            print(f"[{filepath}:{line_num}] Warning: Evaluated line is not a dictionary: {type(parsed)}")
            return None
    except Exception as e:
        print(f"[{filepath}:{line_num}] Error parsing line: {e}")
        return None

def recover_log_file(log_path, csv_path):
    print(f"Processing: {log_path} -> {csv_path}")
    rows = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            parsed = parse_log_line(line, idx, log_path)
            if parsed is not None:
                rows.append(parsed)
    
    if not rows:
        print(f"  No valid rows found in {log_path}. Skipping CSV creation.")
        return False
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"  Successfully wrote {len(rows)} rows to {csv_path}")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recover CSV files from interrupted experiment or grid search log files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV files.")
    args = parser.parse_args()

    reports_dir = "reports"
    logs_dir = os.path.join(reports_dir, "logs")

    # 1. Scan for grid_search logs in reports/logs/
    # grid_search_log_<timestamp>.txt -> reports/average_run/grid_search_results_<timestamp>.csv
    if os.path.exists(logs_dir):
        for filename in sorted(os.listdir(logs_dir)):
            if filename.startswith("grid_search_log_") and filename.endswith(".txt"):
                match = re.match(r"grid_search_log_(.*)\.txt", filename)
                if match:
                    timestamp = match.group(1)
                    log_path = os.path.join(logs_dir, filename)
                    csv_path = os.path.join(reports_dir, "average_run", f"grid_search_results_{timestamp}.csv")
                    
                    if os.path.exists(csv_path) and not args.overwrite:
                        continue
                    recover_log_file(log_path, csv_path)

            # 2. Scan for experiment logs in reports/logs/
            # experiment_log_<timestamp>.txt -> reports/single_run/experiment_results_<timestamp>.csv
            elif filename.startswith("experiment_log_") and filename.endswith(".txt"):
                match = re.match(r"experiment_log_(.*)\.txt", filename)
                if match:
                    timestamp = match.group(1)
                    log_path = os.path.join(logs_dir, filename)
                    csv_path = os.path.join(reports_dir, "single_run", f"experiment_results_{timestamp}.csv")
                    
                    if os.path.exists(csv_path) and not args.overwrite:
                        continue
                    recover_log_file(log_path, csv_path)

    # 3. Scan for experiment logs in reports/ (top-level)
    # experiment_log_<timestamp>.txt -> reports/experiment_results_<timestamp>.csv
    if os.path.exists(reports_dir):
        for filename in sorted(os.listdir(reports_dir)):
            if filename.startswith("experiment_log_") and filename.endswith(".txt"):
                match = re.match(r"experiment_log_(.*)\.txt", filename)
                if match:
                    timestamp = match.group(1)
                    log_path = os.path.join(reports_dir, filename)
                    csv_path = os.path.join(reports_dir, f"experiment_results_{timestamp}.csv")
                    
                    if os.path.exists(csv_path) and not args.overwrite:
                        continue
                    recover_log_file(log_path, csv_path)

if __name__ == "__main__":
    main()
