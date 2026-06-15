import os
import pickle
from collections import defaultdict
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load credentials from .env
# Adjust path if necessary
load_dotenv("/home/cambria/MineGraphRule/GRAM/.env")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:37687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mineGraphRule")

BASE_DIR = "/home/cambria/MineGraphRule/ClassificationforMineGraphRule/kge"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/custom_dataset")

def sanitize(text):
    if text is None:
        return "None"
    s = str(text).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
    return s if s else "EmptyString"

def prepare_kgreasoning_data(triples, entity_dict, relation_dict):
    train_queries = defaultdict(set)
    train_answers = defaultdict(set)
    
    # 1p queries: (h, (r,)) -> {t}
    q_structure = ('e', ('r',))
    for h, r, t in triples:
        h_idx = entity_dict[h]
        r_idx = relation_dict[r]
        t_idx = entity_dict[t]
        query = (h_idx, (r_idx,))
        train_queries[q_structure].add(query)
        train_answers[query].add(t_idx)
        
    # Convert sets to lists for pickling
    train_queries_final = {k: list(v) for k, v in train_queries.items()}
    
    with open(os.path.join(OUTPUT_DIR, "train-queries.pkl"), "wb") as f:
        pickle.dump(train_queries_final, f)
    with open(os.path.join(OUTPUT_DIR, "train-answers.pkl"), "wb") as f:
        pickle.dump(train_answers, f)
        
    # Also create valid/test empty shells
    for split in ['valid', 'test']:
        with open(os.path.join(OUTPUT_DIR, f"{split}-queries.pkl"), "wb") as f:
            pickle.dump({}, f)
        with open(os.path.join(OUTPUT_DIR, f"{split}-hard-answers.pkl"), "wb") as f:
            pickle.dump({}, f)
        with open(os.path.join(OUTPUT_DIR, f"{split}-easy-answers.pkl"), "wb") as f:
            pickle.dump({}, f)

    with open(os.path.join(OUTPUT_DIR, "stats.txt"), "w") as f:
        f.write(f"nentity: {len(entity_dict)}\n")
        f.write(f"nrelation: {len(relation_dict)}\n")

def download_triplets(mode="average", format="txt"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        triples = []
        
        with driver.session() as session:
            # Basic instance triples
            print(f"Fetching instance triples for mode: {mode}")
            query1 = "MATCH (h)-[r]->(t) WHERE h.name IS NOT NULL AND t.name IS NOT NULL RETURN h.name, type(r), t.name"
            result1 = session.run(query1)
            for record in result1:
                triples.append((sanitize(record[0]), sanitize(record[1]), sanitize(record[2])))
                
            if mode in ["labelled", "hybrid"]:
                print(f"Fetching label triples for mode: {mode}")
                query2 = "MATCH (n) WHERE n.name IS NOT NULL UNWIND labels(n) as l RETURN n.name, 'HAS_LABEL', l"
                result2 = session.run(query2)
                for record in result2:
                    triples.append((sanitize(record[0]), "HAS_LABEL", sanitize(record[2])))

        driver.close()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return
    
    print(f"Extracted {len(triples)} triples.")
    
    if not triples:
        print("No triples found!")
        return

    # Create dictionaries
    entities = set()
    relations = set()
    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
        
    entity_list = sorted(list(entities))
    entity_dict = {ent: idx for idx, ent in enumerate(entity_list)}
    with open(os.path.join(OUTPUT_DIR, "entities.dict"), "w") as f:
        for idx, ent in enumerate(entity_list):
            f.write(f"{idx}\t{ent}\n")
            
    relation_list = sorted(list(relations))
    relation_dict = {rel: idx for idx, rel in enumerate(relation_list)}
    with open(os.path.join(OUTPUT_DIR, "relations.dict"), "w") as f:
        for idx, rel in enumerate(relation_list):
            f.write(f"{idx}\t{rel}\n")

    # Save to train.txt
    train_file = os.path.join(OUTPUT_DIR, "train.txt")
    with open(train_file, "w") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    
    # Create empty valid and test files for compatibility
    with open(os.path.join(OUTPUT_DIR, "valid.txt"), "w") as f:
        pass
    with open(os.path.join(OUTPUT_DIR, "test.txt"), "w") as f:
        pass

    if format == "kgw":
        print("Preparing KGReasoning (pkl) files...")
        prepare_kgreasoning_data(triples, entity_dict, relation_dict)

    print(f"Data saved to {OUTPUT_DIR}")
    print(f"Total Entities: {len(entities)}")
    print(f"Total Relations: {len(relations)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="average", choices=["average", "labelled", "hybrid"])
    parser.add_argument("--format", type=str, default="txt", choices=["txt", "kgw"])
    args = parser.parse_args()
    
    download_triplets(mode=args.mode, format=args.format)
