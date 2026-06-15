import pandas as pd
import os
import random

def merge_law_rules():
    source_dir = 'rules/law/single_files'
    output_dir = 'rules/law/merged'
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    all_rules = []

    print(f"Reading rules from {len(csv_files)} files...")
    for filename in csv_files:
        file_path = os.path.join(source_dir, filename)
        try:
            # Using low_memory=False to handle potential type issues in large files
            df = pd.read_csv(file_path, low_memory=False)
            all_rules.append(df)
            print(f"  Loaded {len(df)} rules from {filename}")
        except Exception as e:
            print(f"  Error reading {filename}: {e}")

    if not all_rules:
        print("No rules found!")
        return

    # Combine all DataFrames
    combined_df = pd.concat(all_rules, ignore_index=True)
    
    # Shuffle to ensure variation
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_available = len(combined_df)
    print(f"Total rules available: {total_available}")

    counts = [250, 1000, 5000]
    
    for count in counts:
        if total_available >= count:
            subset_df = combined_df.head(count)
            output_file = os.path.join(output_dir, f"Law_Merged_{count}.csv")
            subset_df.to_csv(output_file, index=False)
            print(f"Saved {count} rules to {output_file}")
        else:
            print(f"Skipping {count} merge - only {total_available} rules available.")

if __name__ == "__main__":
    merge_law_rules()