import pandas as pd
import re
import os
import argparse
from dotenv import load_dotenv

def extract_schema_from_csv(csv_path, sep=';'):
    print(f"Extracting schema from CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep=sep)
    schema_triples = set()
    
    def process_pattern(pattern):
        if not isinstance(pattern, str):
            return
        # Find patterns like (Label1)-[REL]->(Label2)
        matches = re.findall(r'\((.*?)\)-\[(.*?)\]->\((.*?)\)', pattern)
        for head, rel, tail in matches:
            schema_triples.add((head, rel, tail))

    for col in ['Body', 'Head']:
        if col in df.columns:
            df[col].apply(process_pattern)
    
    return schema_triples

def extract_schema_from_neo4j(uri, user, pw):
    print(f"Extracting schema from Neo4j: {uri}")
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    schema_triples = set()
    
    query = """
    MATCH (n)-[r]->(m)
    WITH DISTINCT labels(n)[0] AS head, type(r) AS rel, labels(m)[0] AS tail
    WHERE head IS NOT NULL AND rel IS NOT NULL AND tail IS NOT NULL
    RETURN head, rel, tail
    """
    
    with driver.session() as session:
        results = session.run(query)
        for record in results:
            schema_triples.add((record['head'], record['rel'], record['tail']))
            
    driver.close()
    return schema_triples

def save_schema_dataset(schema, output_dir='kge/data/schema_dataset'):
    os.makedirs(output_dir, exist_ok=True)
    
    entities = sorted(list(set([t[0] for t in schema] + [t[2] for t in schema])))
    relations = sorted(list(set([t[1] for t in schema])))

    with open(os.path.join(output_dir, 'entities.dict'), 'w') as f:
        for i, ent in enumerate(entities):
            f.write(f"{i}\t{ent}\n")

    with open(os.path.join(output_dir, 'relations.dict'), 'w') as f:
        for i, rel in enumerate(relations):
            f.write(f"{i}\t{rel}\n")

    for split in ['train.txt', 'valid.txt', 'test.txt']:
        with open(os.path.join(output_dir, split), 'w') as f:
            for h, r, t in schema:
                f.write(f"{h}\t{r}\t{t}\n")

    print(f"Saved schema dataset to {output_dir}")
    print(f"Total Triples: {len(schema)}")
    print(f"Entities: {entities}")
    print(f"Relations: {relations}")

def main():
    parser = argparse.ArgumentParser(description="Extract schema triples from CSV or Neo4j.")
    parser.add_argument("--csv", help="Input CSV file path.")
    parser.add_argument("--sep", default=";", help="CSV separator (default: ';').")
    parser.add_argument("--neo4j", action="store_true", help="Extract from Neo4j (requires .env with credentials).")
    parser.add_argument("--env", default="/home/cambria/MineGraphRule/GRAM/.env", help="Path to .env file for Neo4j.")
    parser.add_argument("--output", default="kge/data/schema_dataset", help="Output directory for schema dataset.")
    
    args = parser.parse_args()
    
    schema = set()
    if args.neo4j:
        load_dotenv(args.env)
        uri = os.getenv('NEO4J_URI')
        user = os.getenv('NEO4J_USER')
        pw = os.getenv('NEO4J_PASSWORD')
        if not all([uri, user, pw]):
            print("Error: Neo4j credentials not found in .env file.")
            return
        schema = extract_schema_from_neo4j(uri, user, pw)
    elif args.csv:
        schema = extract_schema_from_csv(args.csv, args.sep)
    else:
        # Default behavior: try RulesSpotify/RulesSpotify_Merged.csv
        default_csv = 'RulesSpotify/RulesSpotify_Merged.csv'
        if os.path.exists(default_csv):
            schema = extract_schema_from_csv(default_csv, args.sep)
        else:
            print("No input provided and default CSV not found. Use --csv or --neo4j.")
            return

    if schema:
        save_schema_dataset(schema, args.output)
    else:
        print("No schema triples extracted.")

if __name__ == "__main__":
    main()
