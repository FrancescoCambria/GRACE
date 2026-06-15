import os
import subprocess
import torch
import numpy as np
import json
import pickle
from collections import defaultdict
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv("/home/cambria/MineGraphRule/GRAM/.env")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:37687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mineGraphRule")

BASE_DIR = "/home/cambria/MineGraphRule/ClassificationforMineGraphRule/kge"
DATA_DIR = os.path.join(BASE_DIR, "data/custom_dataset")
KGE_HOME = os.path.join(BASE_DIR, "KnowledgeGraphEmbedding")
KGREASONING_HOME = os.path.join(BASE_DIR, "KGReasoning")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/custom_model")
Q2B_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/q2b_model")

class KGEHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.entity_dict = {}
        self.relation_dict = {}
        self.entity_emb = None
        self.relation_emb = None
        self.model_name = "RotatE"
        self.embedding_range = None

    def close(self):
        self.driver.close()

    def _sanitize(self, text):
        if text is None:
            return "None"
        s = str(text).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
        return s if s else "EmptyString"

    def fetch_triples(self, query):
        print(f"Executing triple extraction query: {query}")
        with self.driver.session() as session:
            result = session.run(query)
            triples = [record.values() for record in result]
            print(f"Extracted {len(triples)} triples.")
            return triples

    def prepare_data(self, triples, model_name="RotatE"):
        os.makedirs(DATA_DIR, exist_ok=True)
        
        entities = set()
        relations = set()
        
        formatted_triples = []
        for head, rel, tail in triples:
            h = self._sanitize(head)
            r = self._sanitize(rel)
            t = self._sanitize(tail)
            entities.add(h)
            entities.add(t)
            relations.add(r)
            formatted_triples.append((h, r, t))
            
        entity_list = sorted(list(entities))
        with open(os.path.join(DATA_DIR, "entities.dict"), "w") as f:
            for idx, entity in enumerate(entity_list):
                f.write(f"{idx}\t{entity}\n")
                self.entity_dict[entity] = idx
                
        relation_list = sorted(list(relations))
        with open(os.path.join(DATA_DIR, "relations.dict"), "w") as f:
            for idx, relation in enumerate(relation_list):
                f.write(f"{idx}\t{relation}\n")
                self.relation_dict[relation] = idx
        
        if model_name == "Query2Box":
            self._prepare_kgreasoning_data(formatted_triples, len(entities), len(relations))
        else:
            with open(os.path.join(DATA_DIR, "train.txt"), "w") as f:
                for h, r, t in formatted_triples:
                    f.write(f"{h}\t{r}\t{t}\n")
            open(os.path.join(DATA_DIR, "valid.txt"), "w").close()
            open(os.path.join(DATA_DIR, "test.txt"), "w").close()
        
        print(f"Data prepared in {DATA_DIR}. {len(entities)} entities, {len(relations)} relations.")

    def _prepare_kgreasoning_data(self, triples, nentity, nrelation):
        train_queries = defaultdict(set)
        train_answers = defaultdict(set)
        
        # 1p queries: (h, (r,)) -> {t}
        q_structure = ('e', ('r',))
        for h, r, t in triples:
            h_idx = self.entity_dict[h]
            r_idx = self.relation_dict[r]
            t_idx = self.entity_dict[t]
            query = (h_idx, (r_idx,))
            train_queries[q_structure].add(query)
            train_answers[query].add(t_idx)
            
        # Convert sets to lists for pickling
        train_queries_final = {k: list(v) for k, v in train_queries.items()}
        
        with open(os.path.join(DATA_DIR, "train-queries.pkl"), "wb") as f:
            pickle.dump(train_queries_final, f)
        with open(os.path.join(DATA_DIR, "train-answers.pkl"), "wb") as f:
            pickle.dump(train_answers, f)
            
        # Also create valid/test empty shells
        for split in ['valid', 'test']:
            with open(os.path.join(DATA_DIR, f"{split}-queries.pkl"), "wb") as f:
                pickle.dump({}, f)
            with open(os.path.join(DATA_DIR, f"{split}-hard-answers.pkl"), "wb") as f:
                pickle.dump({}, f)
            with open(os.path.join(DATA_DIR, f"{split}-easy-answers.pkl"), "wb") as f:
                pickle.dump({}, f)

        with open(os.path.join(DATA_DIR, "stats.txt"), "w") as f:
            f.write(f"nentity: {nentity}\n")
            f.write(f"nrelation: {nrelation}\n")

    def train(self, model_name="RotatE", max_steps=1000, dim=192):
        self.model_name = model_name
        
        if model_name == "Query2Box":
            self._train_query2box(max_steps, dim)
        else:
            self._train_rotate(model_name, max_steps, dim)
            
        self.load_embeddings()

    def _train_rotate(self, model_name, max_steps, dim):
        cmd = [
            "python3", os.path.join(KGE_HOME, "codes", "run.py"),
            "--do_train",
            "--data_path", DATA_DIR,
            "--model", model_name,
            "-n", "256", "-b", "1024", "-d", str(dim),
            "-g", "24.0", "-a", "1.0", "-adv",
            "-lr", "0.0001", "--max_steps", str(max_steps),
            "-save", MODEL_SAVE_PATH,
            "--test_batch_size", "16"
        ]
        if model_name in ["RotatE", "ComplEx"]:
            cmd.append("--double_entity_embedding")
        if model_name == "ComplEx":
            cmd.append("--double_relation_embedding")
        if torch.cuda.is_available():
            cmd.append("--cuda")
            
        print(f"Starting training for {model_name}...")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(KGE_HOME, "codes")
        subprocess.run(cmd, check=True, env=env)
        print(f"Training completed. Model saved to {MODEL_SAVE_PATH}")

    def _train_query2box(self, max_steps, dim):
        os.makedirs(Q2B_MODEL_SAVE_PATH, exist_ok=True)
        # KGReasoning uses a hardcoded logging logic in main.py that uses --prefix
        # We'll set prefix to Q2B_MODEL_SAVE_PATH
        cmd = [
            "python3", os.path.join(KGREASONING_HOME, "main.py"),
            "--do_train",
            "--data_path", DATA_DIR,
            "--geo", "box",
            "-n", "128", "-b", "512", "-d", str(dim),
            "-g", "12.0", "-lr", "0.001", "--max_steps", str(max_steps),
            "--prefix", Q2B_MODEL_SAVE_PATH,
            "--tasks", "1p"
        ]
        if torch.cuda.is_available():
            cmd.append("--cuda")
            
        print(f"Starting training for Query2Box...")
        subprocess.run(cmd, check=True)
        
        # The save path is built as: args.prefix / data_dir_name / tasks / geo / gamma_mode_str / cur_time
        # In our case: Q2B_MODEL_SAVE_PATH / custom_dataset / 1p / box / g-12.0-mode-(none,0.02) / <timestamp>
        data_dir_name = os.path.basename(DATA_DIR)
        save_root = os.path.join(Q2B_MODEL_SAVE_PATH, data_dir_name, "1p", "box", "g-12.0-mode-(none,0.02)")
        
        # Find the subfolder (timestamp based)
        subfolders = sorted([f for f in os.listdir(save_root) if os.path.isdir(os.path.join(save_root, f))])
        if subfolders:
            latest_path = os.path.join(save_root, subfolders[-1])
            self.current_checkpoint_path = os.path.join(latest_path, "checkpoint")
            print(f"Training completed. Model saved to {latest_path}")
        else:
            print(f"Could not find Query2Box checkpoint subfolder in {save_root}")

    def load_embeddings(self):
        if self.model_name == "Query2Box":
            checkpoint_path = self.current_checkpoint_path
        else:
            checkpoint_path = os.path.join(MODEL_SAVE_PATH, "checkpoint")
            
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return

        if not self.entity_dict:
            with open(os.path.join(DATA_DIR, "entities.dict"), "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        idx, name = parts[0], "\t".join(parts[1:])
                        self.entity_dict[name] = int(idx)
        if not self.relation_dict:
            with open(os.path.join(DATA_DIR, "relations.dict"), "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        idx, name = parts[0], "\t".join(parts[1:])
                        self.relation_dict[name] = int(idx)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint['model_state_dict']
        
        self.entity_emb = state_dict['entity_embedding'].numpy()
        self.relation_emb = state_dict['relation_embedding'].numpy()
        if 'offset_embedding' in state_dict:
            self.offset_emb = state_dict['offset_embedding'].numpy()
        
        if 'embedding_range' in state_dict:
            self.embedding_range = state_dict['embedding_range'].item()
        
        print(f"Embeddings loaded successfully. Model: {self.model_name}")

    def get_pattern_embedding(self, pattern_query):
        if self.entity_emb is None:
            self.load_embeddings()

        if not pattern_query.upper().startswith("MATCH"):
            match_query = f"MATCH p = {pattern_query} RETURN p LIMIT 100"
        else:
            match_query = pattern_query

        print(f"Executing pattern query: {match_query}")
        
        instance_embeddings = []
        with self.driver.session() as session:
            result = session.run(match_query)
            for record in result:
                path = record["p"]
                
                if self.model_name == "Query2Box" and self.offset_emb is not None:
                    # GEOMETRIC PROJECTION LOGIC
                    nodes = list(path.nodes)
                    rels = list(path.relationships)
                    if not nodes: continue
                    
                    # Start at the first node
                    start_name = self._sanitize(nodes[0].get("name"))
                    if start_name not in self.entity_dict: continue
                    
                    curr_c = self.entity_emb[self.entity_dict[start_name]]
                    curr_o = np.zeros_like(curr_c)
                    
                    # Iteratively project through the relations
                    for rel in rels:
                        rel_type = self._sanitize(rel.type)
                        if rel_type in self.relation_dict:
                            rel_idx = self.relation_dict[rel_type]
                            curr_c = curr_c + self.relation_emb[rel_idx]
                            # Use Softplus activation for offset as per Q2B standard
                            curr_o = curr_o + np.log(1 + np.exp(self.offset_emb[rel_idx]))
                    
                    # The pattern embedding is the concatenation of center and offset [c, o]
                    instance_embeddings.append(np.concatenate([curr_c, curr_o]))
                
                else:
                    # FALLBACK: MEAN OF COMPONENTS (Used for RotatE or if offsets missing)
                    components = []
                    for node in path.nodes:
                        name = self._sanitize(node.get("name"))
                        if name in self.entity_dict:
                            components.append(self.entity_emb[self.entity_dict[name]])
                    
                    for rel in path.relationships:
                        rel_type = self._sanitize(rel.type)
                        if rel_type in self.relation_dict:
                            rel_emb = self.relation_emb[self.relation_dict[rel_type]]
                            if self.model_name == "RotatE" and self.embedding_range is not None:
                                pi = 3.14159265358979323846
                                phase_relation = rel_emb / (self.embedding_range / pi)
                                re_relation = np.cos(phase_relation)
                                im_relation = np.sin(phase_relation)
                                rel_emb = np.concatenate([re_relation, im_relation])
                            components.append(rel_emb)
                    
                    if components:
                        shapes = [c.shape for c in components]
                        if len(set(shapes)) > 1:
                            max_dim = max(s[0] for s in shapes)
                            padded_components = []
                            for c in components:
                                if c.shape[0] < max_dim:
                                    padded_components.append(np.pad(c, (0, max_dim - c.shape[0])))
                                else:
                                    padded_components.append(c)
                            instance_embeddings.append(np.mean(padded_components, axis=0))
                        else:
                            instance_embeddings.append(np.mean(components, axis=0))

        if not instance_embeddings:
            print("No instances found for the pattern in the database.")
            return None
        
        return np.mean(instance_embeddings, axis=0)

    def fetch_triples_for_mode(self, mode="average"):
        """
        Fetches triples from Neo4j based on the training mode.
        - average: Standard instance-level triples.
        - labelled: Instance-level triples + (Instance)-[:HAS_LABEL]->(LabelNode).
        - hybrid: Same as labelled (can be extended if needed).
        """
        triples = []
        # Basic instance triples
        print(f"Fetching instance triples for mode: {mode}")
        q1 = "MATCH (h)-[r]->(t) WHERE h.name IS NOT NULL AND t.name IS NOT NULL RETURN h.name, type(r), t.name"
        triples.extend(self.fetch_triples(q1))
        
        if mode in ["labelled", "hybrid"]:
            print(f"Fetching label triples for mode: {mode}")
            # Add label relationships as virtual triples
            q2 = "MATCH (n) WHERE n.name IS NOT NULL UNWIND labels(n) as l RETURN n.name, 'HAS_LABEL', l"
            triples.extend(self.fetch_triples(q2))
            
        return triples

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="average", choices=["average", "labelled", "hybrid"])
    parser.add_argument("--model", type=str, default="Query2Box", choices=["Query2Box", "RotatE", "TransE", "DistMult", "ComplEx"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=192)
    args = parser.parse_args()

    handler = KGEHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    triples = handler.fetch_triples_for_mode(mode=args.mode)
    
    if triples:
        print(f"Preparing data for {args.model} in {args.mode} mode...")
        handler.prepare_data(triples, model_name=args.model)
        print(f"Starting {args.model} training ({args.steps} steps)...")
        handler.train(model_name=args.model, max_steps=args.steps, dim=args.dim)
        
        # Save a marker for the mode
        save_root = Q2B_MODEL_SAVE_PATH if args.model == "Query2Box" else MODEL_SAVE_PATH
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, "current_mode.txt"), "w") as f:
            f.write(args.mode)

    handler.close()
