import os
import pickle
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load credentials
load_dotenv("/home/cambria/MineGraphRule/GRAM/.env")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:37687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mineGraphRule")

def sanitize(text):
    if text is None: return "None"
    return str(text).replace("\t", " ").replace("\n", " ").strip()

def export_all_triples(output_path="kge/all_triples.pkl"):
    print(f"Connecting to {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        triples = []
        
        with driver.session() as session:
            print("Fetching instance triples...")
            query1 = "MATCH (h)-[r]->(t) WHERE h.name IS NOT NULL AND t.name IS NOT NULL RETURN h.name, type(r), t.name"
            result1 = session.run(query1)
            for record in result1:
                triples.append((sanitize(record[0]), sanitize(record[1]), sanitize(record[2])))
                
            print("Fetching label triples...")
            query2 = "MATCH (n) WHERE n.name IS NOT NULL UNWIND labels(n) as l RETURN n.name, 'HAS_LABEL', l"
            result2 = session.run(query2)
            for record in result2:
                triples.append((sanitize(record[0]), "HAS_LABEL", sanitize(record[2])))
        
        driver.close()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return
    
    print(f"Extracted {len(triples)} triples.")
    
    # Save as pickle for easy loading in Python
    with open(output_path, "wb") as f:
        pickle.dump(triples, f)
    
    # Also save as a simple text file for visibility
    txt_path = output_path.replace(".pkl", ".txt")
    with open(txt_path, "w") as f:
        for head, rel, tail in triples:
            f.write(f"{head}\t{rel}\t{tail}\n")
            
    print(f"Saved to {output_path} and {txt_path}")

if __name__ == "__main__":
    export_all_triples()
