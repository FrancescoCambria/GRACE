import os
import csv
import re
import pickle
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load credentials from project root
# Assuming the script is in src/utils/maintenance/
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(os.path.join(base_dir, '.env'))

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
pw = os.getenv('NEO4J_PASSWORD')

# Cache for ID -> (Label, Name)
CACHE_FILE = "law_id_info_cache.pkl"

def get_node_info(node_id, driver, cache):
    if not node_id: return None, None
    if node_id in cache:
        return cache[node_id]
    
    with driver.session() as session:
        # Law IDs are internal Neo4j IDs (integers)
        query = "MATCH (n) WHERE ID(n) = toInteger($node_id) RETURN labels(n)[0] as label, n.name as name, n.title as title"
        try:
            res = session.run(query, node_id=node_id)
            record = res.single()
            if record:
                label = record['label']
                name = record['name'] or record['title'] or f"Node_{node_id}"
                cache[node_id] = (label, name)
                return label, name
        except:
            # Fallback for elementId
            query_eid = "MATCH (n) WHERE elementId(n) = $node_id RETURN labels(n)[0] as label, n.name as name, n.title as title"
            try:
                res = session.run(query_eid, node_id=node_id)
                record = res.single()
                if record:
                    label = record['label']
                    name = record['name'] or record['title'] or f"Node_{node_id}"
                    cache[node_id] = (label, name)
                    return label, name
            except:
                pass
    
    return None, f"Unknown_{node_id}"

def extract_last_label(pattern):
    labels = re.findall(r"\(([^)]+)\)", pattern)
    if labels:
        return labels[-1]
    return None

def fix_mismatch(patterns_str, ids_str, driver, cache):
    PATTERN_SEP = "; "
    ID_SEP = ";"
    
    patterns = [p.strip() for p in patterns_str.split(PATTERN_SEP) if p.strip()]
    ids = [idx.strip() for idx in ids_str.split(ID_SEP) if idx.strip()]
    
    if not ids:
        return patterns_str, ids_str, ""

    infos = [get_node_info(idx, driver, cache) for idx in ids]
    id_labels = [info[0] for info in infos]
    id_names = [info[1] for info in infos]
    
    if len(patterns) == 1 and len(ids) > 1:
        new_patterns = [patterns[0]] * len(ids)
        return PATTERN_SEP.join(new_patterns), ids_str, ID_SEP.join(id_names)

    pattern_labels = [extract_last_label(p) for p in patterns]
    
    label_to_items = {}
    for idx, label, name in zip(ids, id_labels, id_names):
        if label not in label_to_items: label_to_items[label] = []
        label_to_items[label].append((idx, name))
    
    label_patterns_indices = {}
    for i, l in enumerate(pattern_labels):
        if l not in label_patterns_indices: label_patterns_indices[l] = []
        label_patterns_indices[l].append(i)
        
    final_mapping = [None] * len(patterns)
    for label, p_indices in label_patterns_indices.items():
        items = label_to_items.get(label, [])
        if not items: continue
        
        n_p = len(p_indices)
        n_i = len(items)
        base_items_per_p = n_i // n_p
        remainder = n_i % n_p
        
        curr_i = 0
        for i, p_idx in enumerate(p_indices):
            count = base_items_per_p + (1 if i < remainder else 0)
            final_mapping[p_idx] = items[curr_i : curr_i + count]
            curr_i += count

    new_patterns = []
    new_ids = []
    new_names = []
    for p, mapping in zip(patterns, final_mapping):
        if mapping:
            for idx, name in mapping:
                new_patterns.append(p)
                new_ids.append(idx)
                new_names.append(name)
        else:
            new_patterns.append(p)
            new_ids.append("")
            new_names.append("")

    # Filter out empty entries if we have at least some valid ones
    if any(new_ids):
        valid_indices = [i for i, val in enumerate(new_ids) if val]
        new_patterns = [new_patterns[i] for i in valid_indices]
        new_ids = [new_ids[i] for i in valid_indices]
        new_names = [new_names[i] for i in valid_indices]

    if not new_ids:
        return patterns_str, ids_str, ID_SEP.join(id_names)

    return PATTERN_SEP.join(new_patterns), ID_SEP.join(new_ids), ID_SEP.join(new_names)

def process_law_files(input_dir="rules/law/tagged"):
    if not os.path.exists(input_dir): 
        print(f"Directory not found: {input_dir}")
        return

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f: cache = pickle.load(f)
    else: cache = {}

    driver = GraphDatabase.driver(uri, auth=(user, pw))
    try:
        files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
        for filename in files:
            file_path = os.path.join(input_dir, filename)
            print(f"Processing {file_path}...")
            rows = []
            
            # Detect separator
            with open(file_path, "r", encoding="utf-8") as f:
                header = f.readline()
                sep = ";" if ";" in header else ","
            
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=sep)
                fieldnames = list(reader.fieldnames)
                if "Body Node Names" not in fieldnames:
                    if "Body Node IDs" in fieldnames:
                        fieldnames.insert(fieldnames.index("Body Node IDs") + 1, "Body Node Names")
                    else: fieldnames.append("Body Node Names")
                if "Head Node Names" not in fieldnames:
                    if "Head Node IDs" in fieldnames:
                        fieldnames.insert(fieldnames.index("Head Node IDs") + 1, "Head Node Names")
                    else: fieldnames.append("Head Node Names")
                
                for row in reader:
                    row["Body"], row["Body Node IDs"], row["Body Node Names"] = fix_mismatch(
                        row.get("Body", ""), row.get("Body Node IDs", ""), driver, cache
                    )
                    row["Head"], row["Head Node IDs"], row["Head Node Names"] = fix_mismatch(
                        row.get("Head", ""), row.get("Head Node IDs", ""), driver, cache
                    )
                    rows.append(row)
            
            # Save updated file with semicolon delimiter
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)
            print(f"Successfully processed {filename}")
    finally:
        driver.close()
        with open(CACHE_FILE, "wb") as f: pickle.dump(cache, f)

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "rules/law/tagged"
    process_law_files(target_dir)
