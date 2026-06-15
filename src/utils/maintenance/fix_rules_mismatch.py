import os
import csv
import re
import pickle
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load credentials
load_dotenv('/home/cambria/MineGraphRule/GRAM/.env')
uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
pw = os.getenv('NEO4J_PASSWORD')

# Cache for ID -> Label
LABEL_CACHE_FILE = "id_label_cache.pkl"

def get_label(node_id, driver, label_cache):
    if node_id in label_cache:
        return label_cache[node_id]
    
    with driver.session() as session:
        # Using elementId for Neo4j compatibility
        res = session.run("MATCH (n) WHERE elementId(n) = $node_id RETURN labels(n)[0] as label", node_id=node_id)
        record = res.single()
        if record:
            label = record['label']
            label_cache[node_id] = label
            return label
    return None

def extract_last_label(pattern):
    # Find all occurrences of (Label)
    labels = re.findall(r"\(([^)]+)\)", pattern)
    if labels:
        return labels[-1]
    return None

def fix_mismatch(patterns_str, ids_str, names_str, driver, label_cache):
    patterns = [p.strip() for p in patterns_str.split(", ") if p.strip()]
    ids = [idx.strip() for idx in ids_str.split(",") if idx.strip()]
    names = [n.strip() for n in names_str.split(",") if n.strip()]
    
    if not ids:
        return patterns_str, ids_str, names_str

    # 1) if there is only one graph pattern and multiple IDs replicate in the body or head column
    if len(patterns) == 1 and len(ids) > 1:
        new_patterns = [patterns[0]] * len(ids)
        return ", ".join(new_patterns), ids_str, names_str

    # 2) match the ids in the ID column and check if their label matches the last label of the graph pattern
    # 3) check order of the labels and reorder accordingly
    id_labels = [get_label(idx, driver, label_cache) for idx in ids]
    pattern_labels = [extract_last_label(p) for p in patterns]
    
    # Map labels to IDs and Names (keeping multiples for the same label)
    label_to_items = {}
    for idx, name, label in zip(ids, names, id_labels):
        if label not in label_to_items:
            label_to_items[label] = []
        label_to_items[label].append((idx, name))
    
    new_patterns = []
    new_ids = []
    new_names = []
    
    # Iterate through patterns and pick corresponding IDs to maintain pattern order
    for p, p_label in zip(patterns, pattern_labels):
        if p_label in label_to_items and label_to_items[p_label]:
            # Take all IDs that match this label for this pattern instance
            # Note: This logic assumes if multiple IDs match the same label, 
            # and that label appears once in the pattern list, they all belong to it.
            for idx, name in label_to_items[p_label]:
                new_patterns.append(p)
                new_ids.append(idx)
                new_names.append(name)
            # Clear them so they are not reused if labels are unique per pattern sequence
            label_to_items[p_label] = []

    # If the logic above didn't account for all IDs (mismatch in count), we keep original or return fixed
    if not new_ids:
        return patterns_str, ids_str, names_str

    return ", ".join(new_patterns), ",".join(new_ids), ",".join(new_names)

def process_files():
    input_dir = "RulesSpotify/LLMLogic"
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found.")
        return

    # Load cache
    if os.path.exists(LABEL_CACHE_FILE):
        with open(LABEL_CACHE_FILE, "rb") as f:
            label_cache = pickle.load(f)
    else:
        label_cache = {}

    driver = GraphDatabase.driver(uri, auth=(user, pw))
    
    try:
        files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
        for filename in files:
            file_path = os.path.join(input_dir, filename)
            print(f"Fixing mismatches in {file_path}...")
            
            rows = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                fieldnames = reader.fieldnames
                for row in reader:
                    # Fix Body
                    row["Body"], row["Body Node IDs"], row["Body Node Names"] = fix_mismatch(
                        row["Body"], row["Body Node IDs"], row["Body Node Names"], driver, label_cache
                    )
                    # Fix Head
                    row["Head"], row["Head Node IDs"], row["Head Node Names"] = fix_mismatch(
                        row["Head"], row["Head Node IDs"], row["Head Node Names"], driver, label_cache
                    )
                    rows.append(row)
            
            # Save updated file
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)
    finally:
        driver.close()
        # Save cache
        with open(LABEL_CACHE_FILE, "wb") as f:
            pickle.dump(label_cache, f)

if __name__ == "__main__":
    process_files()
