import os
import csv
import re
import pickle
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

# Hardcoded for Spotify dataset
uri = 'bolt://localhost:23008'
user = 'neo4j'
pw = 'mineGraphRule'

# Cache for ID -> Label
LABEL_CACHE_FILE = "id_label_cache.pkl"

def get_label(node_id, driver, label_cache):
    if node_id in label_cache:
        return label_cache[node_id]
    
    with driver.session() as session:
        try:
            node_id_int = int(node_id)
            res = session.run("MATCH (n) WHERE id(n) = $node_id RETURN labels(n)[0] as label", node_id=node_id_int)
        except ValueError:
            res = session.run("MATCH (n) WHERE elementId(n) = $node_id RETURN labels(n)[0] as label", node_id=node_id)
            
        record = res.single()
        if record:
            label = record['label']
            label_cache[node_id] = label
            return label
    return None

def extract_last_label(pattern):
    labels = re.findall(r"\(([^)]+)\)", pattern)
    if labels:
        return labels[-1]
    return None

def fix_mismatch(patterns_str, ids_str, names_str, driver, label_cache):
    if not patterns_str or not ids_str or not names_str:
        return patterns_str, ids_str, names_str
        
    patterns = [p.strip() for p in patterns_str.split(", ") if p.strip()]
    id_sep = ";" if ";" in ids_str else ","
    name_sep = ";" if ";" in names_str else ","
    
    ids = [idx.strip() for idx in ids_str.split(id_sep) if idx.strip()]
    names = [n.strip() for n in names_str.split(name_sep) if n.strip()]
    
    if len(patterns) == len(ids):
        return patterns_str, ids_str, names_str
    
    label_to_items = {}
    for idx, name in zip(ids, names):
        label = get_label(idx, driver, label_cache)
        if label not in label_to_items:
            label_to_items[label] = []
        label_to_items[label].append((idx, name))
    
    new_patterns = []
    new_ids = []
    new_names = []
    
    for p in patterns:
        p_label = extract_last_label(p)
        if p_label in label_to_items and label_to_items[p_label]:
            for idx, name in label_to_items[p_label]:
                new_patterns.append(p)
                new_ids.append(idx)
                new_names.append(name)
            label_to_items[p_label] = []

    if not new_ids:
        return patterns_str, ids_str, names_str

    return ", ".join(new_patterns), id_sep.join(new_ids), name_sep.join(new_names)

def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    if os.path.exists(LABEL_CACHE_FILE):
        with open(LABEL_CACHE_FILE, "rb") as f:
            label_cache = pickle.load(f)
    else:
        label_cache = {}

    auth = basic_auth(user, pw) if user and pw else None
    driver = GraphDatabase.driver(uri, auth=auth)
    
    try:
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
        
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print("Done.")
    finally:
        driver.close()
        with open(LABEL_CACHE_FILE, "wb") as f:
            pickle.dump(label_cache, f)

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "rules/spotify_5000/RulesSpotify_Merged_5000.csv"
    process_file(target)
