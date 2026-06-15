import os
import pandas as pd
import sys
import argparse

from src.utils.utils import calculate_rule_type_distribution, calculate_node_metrics

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics for all datasets.")
    parser.add_argument("--input_dir", default="RulesSpotify/TaggedDatasets", help="Directory containing datasets.")
    parser.add_argument("--output_file", default="reports/all_datasets_metrics.csv", help="Output CSV file.")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".csv")]
    results = []

    for f in files:
        print(f"Calculating metrics for {f}...")
        df = pd.read_csv(os.path.join(args.input_dir, f), sep=';')
        
        _, total_instances = calculate_rule_type_distribution(df)
        avg_participation, num_unique_nodes = calculate_node_metrics(df)
        
        results.append({
            'Dataset': f,
            'Rule Instances': total_instances,
            'Unique Nodes': num_unique_nodes,
            'Avg Node Participation': avg_participation
        })

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(args.output_file, index=False)
    print(f"Metrics saved to {args.output_file}")

if __name__ == "__main__":
    main()
