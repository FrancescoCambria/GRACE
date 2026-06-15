import pandas as pd
import glob
import os
import argparse
from neo4j import GraphDatabase

def get_node_name(tx, node_id):
    query = """
    MATCH (n)
    WHERE elementId(n) = $node_id
    RETURN labels(n) as labels, n.name as name, n.id as id
    """
    result = tx.run(query, node_id=node_id)
    record = result.single()
    if record:
        labels = record["labels"]
        if "User" in labels:
            return record["id"]
        return record["name"]
    return None

def translate_ids(driver, ids_str):
    if not ids_str or pd.isna(ids_str):
        return ""
    ids = ids_str.split(';')
    names = []
    with driver.session() as session:
        for node_id in ids:
            name = session.execute_read(get_node_name, node_id)
            if name:
                names.append(str(name))
            else:
                names.append(f"Unknown({node_id})")
    return ";".join(names)

def merge_spotify(input_glob, output_file, neo4j_uri, neo4j_auth):
    csv_files = glob.glob(input_glob)
    if not csv_files:
        print(f"No files found for glob: {input_glob}")
        return
        
    all_dfs = []
    for f in csv_files:
        print(f"Reading {f}...")
        df = pd.read_csv(f)
        all_dfs.append(df)
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    print(f"Removed {initial_count - len(merged_df)} duplicate rules.")
    
    merged_df['rule_type'] = pd.factorize(merged_df['Body'] + " | " + merged_df['Head'])[0]
    
    if neo4j_uri and neo4j_auth:
        print("Connecting to Neo4j for ID translation...")
        driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        id_cache = {}
        
        def cached_translate(ids_str):
            if not ids_str or pd.isna(ids_str): return ""
            if ids_str in id_cache: return id_cache[ids_str]
            translated = translate_ids(driver, ids_str)
            id_cache[ids_str] = translated
            return translated

        print("Translating Node IDs...")
        merged_df['Body Node Names'] = merged_df['Body Node IDs'].apply(cached_translate)
        merged_df['Head Node Names'] = merged_df['Head Node IDs'].apply(cached_translate)
        driver.close()
    
    cols = ['rule_type'] + [c for c in merged_df.columns if c != 'rule_type']
    merged_df = merged_df[cols]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True) if os.path.dirname(output_file) else None
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully merged {len(csv_files)} files into {output_file}")

def merge_law(input_dir, output_dir, counts=[250, 1000, 5000]):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
        
    all_rules = []
    for filename in csv_files:
        file_path = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            all_rules.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    combined_df = pd.concat(all_rules, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    total_available = len(combined_df)
    
    for count in counts:
        if total_available >= count:
            subset_df = combined_df.head(count)
            output_file = os.path.join(output_dir, f"Law_Merged_{count}.csv")
            subset_df.to_csv(output_file, index=False)
            print(f"Saved {count} rules to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Unified rule merging script.")
    parser.add_argument("--domain", choices=["spotify", "law"], required=True, help="Domain to merge.")
    parser.add_argument("--input", required=True, help="Input glob (Spotify) or directory (Law).")
    parser.add_argument("--output", required=True, help="Output file (Spotify) or directory (Law).")
    parser.add_argument("--neo4j", action="store_true", help="Enable Neo4j ID translation (Spotify only).")
    parser.add_argument("--uri", default="bolt://localhost:37686", help="Neo4j URI.")
    parser.add_argument("--user", default="neo4j", help="Neo4j username.")
    parser.add_argument("--password", default="mineGraphRule", help="Neo4j password.")
    
    args = parser.parse_args()
    
    if args.domain == "spotify":
        auth = (args.user, args.password) if args.neo4j else None
        merge_spotify(args.input, args.output, args.uri if args.neo4j else None, auth)
    else:
        merge_law(args.input, args.output)

if __name__ == "__main__":
    main()
