import os
import json
import time
import numpy as np
from tqdm import tqdm
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

def get_gemini_embeddings(texts, api_key, model="gemini-embedding-001", cache_file=None):
    client = genai.Client(api_key=api_key)
    cache = {}
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
    
    unique_texts = sorted(list(set([t for t in texts if t not in cache])))
    
    if unique_texts:
        print(f"Embedding {len(unique_texts)} new unique texts...")
        batch_size = 100
        for i in tqdm(range(0, len(unique_texts), batch_size), desc="Gemini Embeddings"):
            batch = unique_texts[i:i+batch_size]
            success = False
            retries = 0
            while not success and retries < 10:
                try:
                    result = client.models.embed_content(
                        model=model, 
                        contents=batch, 
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=384
                        )
                    )
                    for j, t in enumerate(batch):
                        cache[t] = result.embeddings[j].values
                    success = True
                    time.sleep(1.0) 
                except Exception as e:
                    if "429" in str(e):
                        wait_time = (2 ** retries) + 15
                        print(f"Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        print(f"Error in Gemini batch {i}: {e}. Retrying individually...")
                        for t in batch:
                            try:
                                res = client.models.embed_content(
                                    model=model, 
                                    contents=t, 
                                    config=types.EmbedContentConfig(
                                        task_type="RETRIEVAL_DOCUMENT",
                                        output_dimensionality=384
                                    )
                                )
                                cache[t] = res.embeddings[0].values
                                time.sleep(0.5)
                            except:
                                cache[t] = [0.0] * 384
                        success = True
            
            if cache_file:
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)

    return np.array([cache[t] for t in texts])

def get_st_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
