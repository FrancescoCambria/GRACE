import os
import numpy as np
from kge_training import KGEHandler, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BASE_DIR

def main():
    # Hardcoded pattern for the embedding computation
    pattern = "(:Artist)-[:SING]->(:Song)"
    
    handler = KGEHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    print(f"Loading trained model and computing embedding for pattern: {pattern}")
    try:
        emb = handler.get_pattern_embedding(pattern)
        
        if emb is not None:
            print(f"\nSuccess!")
            print(f"Pattern embedding shape: {emb.shape}")
            print(f"Sample values (first 10):")
            print(emb[:10])
            
            # Optional: Save to a specific file
            output_file = os.path.join(BASE_DIR, "latest_pattern_embedding.npy")
            np.save(output_file, emb)
            print(f"\nEmbedding saved to: {output_file}")
        else:
            print("\nCould not compute embedding. Ensure the pattern exists in the Neo4j database.")
            
    finally:
        handler.close()

if __name__ == "__main__":
    main()
