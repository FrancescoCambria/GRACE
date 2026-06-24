import os
import pandas as pd
import numpy as np
import argparse
import torch
import csv
import io
import re
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

# Re-use translation helper
from src.utils.translation import translate_rule

def main():
    parser = argparse.ArgumentParser(description="Generate ST, RotatE, and Combined embeddings for baseline models.")
    parser.add_argument("--input", nargs="+", required=True, help="Input datasets (.csv)")
    parser.add_argument("--dataset", choices=["law", "spotify"], required=True, help="Target dataset (determines Memgraph/Neo4j port and checkpoints)")
    parser.add_argument("--entities_dict", help="Path to entities.dict")
    parser.add_argument("--checkpoint", help="Path to checkpoint")
    parser.add_argument("--kge_hidden_dim", type=int, default=192, help="KGE hidden dimension.")
    parser.add_argument("--st_model_name", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    parser.add_argument("--output_dir", default="rules/embedded_baselines/", help="Output directory for generated datasets.")
    
    args = parser.parse_args()
    
    # Dataset-specific defaults
    port = None
    if args.dataset == "spotify":
        port = 23008
        if not args.entities_dict:
            args.entities_dict = "kge/data/spotify_dataset/entities.dict"
        if not args.checkpoint:
            args.checkpoint = "kge/models/spotify_model/checkpoint"
    elif args.dataset == "law":
        port = 37688
        if not args.entities_dict:
            args.entities_dict = "kge/data/law_dataset/entities.dict"
        if not args.checkpoint:
            args.checkpoint = "kge/models/law_model/checkpoint"

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load KGE Dictionary & Embeddings
    print("Loading KGE dictionaries and checkpoints...")
    entity_dict = {}
    if args.entities_dict and os.path.exists(args.entities_dict):
        with open(args.entities_dict, 'r') as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 2: entity_dict[p[1]] = int(p[0])
    
    entity_emb_matrix = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location='cpu')['model_state_dict']
        if 'entity_embedding' in state:
            entity_emb_matrix = state['entity_embedding'].cpu().numpy()
            print(f"Loaded KGE embeddings from {args.checkpoint}, shape: {entity_emb_matrix.shape}")

    if entity_emb_matrix is None:
        raise ValueError(f"Could not load KGE checkpoint embedding from {args.checkpoint}")

    # 2. Setup Connection Uri & Credentials
    neo4j_env_path = '/home/cambria/gram3/ClassificationforMineGraphRule/.env'
    if os.path.exists(neo4j_env_path):
        load_dotenv(neo4j_env_path)
    else:
        load_dotenv('.env')

    # Force port configuration based on dataset
    if port:
        os.environ["NEO4J_URI"] = f"bolt://localhost:{port}"
        
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USER')
    pw = os.getenv('NEO4J_PASSWORD')
    
    if args.dataset == "law":
        user = ""
        pw = ""
    else:
        if not user:
            user = "neo4j"
        if not pw:
            pw = "mineGraphRule"

    print(f"Connecting to database at {uri} (user: {user})...")
    if user == "" and pw == "":
        auth = None
    else:
        auth = basic_auth(user, pw)
        
    driver = GraphDatabase.driver(uri, auth=auth)
    driver.verify_connectivity()

    # 3. Load SentenceTransformer
    print("Loading SentenceTransformer model...")
    st_model = SentenceTransformer(args.st_model_name)

    # 4. Helper to compute pattern embedding
    def get_pattern_embedding(pattern_str, names_str, session):
        parts = [p.strip() for p in pattern_str.split(',')]
        
        if isinstance(names_str, str):
            f = io.StringIO(names_str)
            reader = csv.reader(f, delimiter=',', quotechar='"')
            try:
                names_list = next(reader, [])
            except StopIteration:
                names_list = []
        else:
            names_list = []
        
        path_embeddings = []
        for i, part in enumerate(parts):
            labels = re.findall(r'\((.*?)\)', part)
            rels = re.findall(r'-\[(.*?)\]->', part)
            if not labels: continue
            
            match_clause = "MATCH p="
            node_clauses = []
            for j, label in enumerate(labels):
                node_clauses.append(f"(n{j}:{label})")
            
            for j in range(len(rels)):
                match_clause += node_clauses[j] + f"-[:{rels[j]}]->"
            match_clause += node_clauses[-1]
            
            query = f"{match_clause} RETURN p LIMIT 50"
            instance_embeddings = []
            try:
                result = session.run(query)
                for record in result:
                    path = record["p"]
                    nodes = list(path.nodes)
                    node_vectors = []
                    for node in nodes:
                        potential_keys = []
                        if "name" in node: potential_keys.append(node["name"])
                        if "title" in node: potential_keys.append(node["title"])
                        potential_keys.append(str(node.id))
                        
                        for key in potential_keys:
                            if key in entity_dict:
                                node_vectors.append(entity_emb_matrix[entity_dict[key]])
                                break
                    if node_vectors:
                        instance_embeddings.append(np.mean(node_vectors, axis=0))
            except Exception as e:
                pass
            
            if instance_embeddings:
                path_embeddings.append(np.mean(instance_embeddings, axis=0))
            else:
                path_embeddings.append(np.zeros(2 * args.kge_hidden_dim))
        
        if path_embeddings:
            return np.mean(path_embeddings, axis=0)
        else:
            return np.zeros(2 * args.kge_hidden_dim)

    # 5. Process each input file
    for input_file in args.input:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        print(f"\nProcessing file: {input_file}")
        
        df = pd.read_csv(input_file, sep=';')
        if len(df.columns) <= 1:
            df = pd.read_csv(input_file, sep=',')
            
        essential_cols = ['Anchor Label', 'Body', 'Head']
        df = df.dropna(subset=essential_cols)
        
        if 'Body Node Names' not in df.columns:
            df['Body Node Names'] = df['Body Node IDs'] if 'Body Node IDs' in df.columns else ""
        if 'Head Node Names' not in df.columns:
            df['Head Node Names'] = df['Head Node IDs'] if 'Head Node IDs' in df.columns else ""
            
        df['Body Node Names'] = df['Body Node Names'].fillna("")
        df['Head Node Names'] = df['Head Node Names'].fillna("")

        # Translate rules
        print("Translating rules to natural language...")
        if 'Natural Language' not in df.columns:
            df['Natural Language'] = df.apply(translate_rule, axis=1)

        # Compute ST Embeddings
        print("Computing text (ST) embeddings...")
        texts = df['Natural Language'].tolist()
        st_embs = st_model.encode(texts, show_progress_bar=True)

        # Compute RotatE Embeddings
        print("Querying graph database for RotatE embeddings...")
        rotate_embs = []
        with driver.session() as session:
            for i in range(len(df)):
                body_emb = get_pattern_embedding(df.iloc[i]['Body'], df.iloc[i]['Body Node Names'], session)
                head_emb = get_pattern_embedding(df.iloc[i]['Head'], df.iloc[i]['Head Node Names'], session)
                # Concatenate body and head embeddings to represent the full rule pattern
                rule_kge_emb = np.concatenate([body_emb, head_emb])
                rotate_embs.append(rule_kge_emb)
        rotate_embs = np.array(rotate_embs)

        # Output ST version
        df_st = df.copy()
        df_st['Embedding'] = [emb.tolist() for emb in st_embs]
        out_st_path = os.path.join(args.output_dir, f"{base_name}_ST.csv")
        df_st.to_csv(out_st_path, sep=';', index=False)
        print(f"Saved ST embeddings to: {out_st_path}")

        # Output RotatE version
        df_rotate = df.copy()
        df_rotate['Embedding'] = [emb.tolist() for emb in rotate_embs]
        out_rotate_path = os.path.join(args.output_dir, f"{base_name}_RotatE.csv")
        df_rotate.to_csv(out_rotate_path, sep=';', index=False)
        print(f"Saved RotatE embeddings to: {out_rotate_path}")

        # Output Combined version
        df_combined = df.copy()
        combined_embs = [np.concatenate([st, rot]) for st, rot in zip(st_embs, rotate_embs)]
        df_combined['Embedding'] = [emb.tolist() for emb in combined_embs]
        out_combined_path = os.path.join(args.output_dir, f"{base_name}_Combined.csv")
        df_combined.to_csv(out_combined_path, sep=';', index=False)
        print(f"Saved Combined embeddings to: {out_combined_path}")

    driver.close()
    print("\nAll embeddings generated successfully!")

if __name__ == "__main__":
    main()
