import os
import pandas as pd
import numpy as np
import argparse
import sys

from src.utils.translation import translate_rule
from src.utils.embedding_engine import get_gemini_embeddings, get_st_embeddings

def main():
    parser = argparse.ArgumentParser(description="Embed rules in datasets.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input datasets.")
    parser.add_argument("--output_dir", required=True, help="Directory to save embedded datasets.")
    parser.add_argument("--type", choices=['gemini', 'st'], required=True, help="Type of embedding to use.")
    parser.add_argument("--api_key", help="Google API key for Gemini.")
    parser.add_argument("--cache_file", default="RulesSpotify/gemini_embeddings_cache.json", help="Cache file for Gemini embeddings.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".csv")]
    all_natural_language_texts = []
    dataset_dfs = {}

    print("Reading datasets and translating rules...")
    for f in files:
        df = pd.read_csv(os.path.join(args.input_dir, f), sep=';')
        if 'Natural Language' not in df.columns:
            df['Natural Language'] = df.apply(translate_rule, axis=1)
        dataset_dfs[f] = df
        all_natural_language_texts.extend(df['Natural Language'].tolist())

    unique_rules = sorted(list(set(all_natural_language_texts)))
    print(f"Total unique rules: {len(unique_rules)}")

    print(f"Generating {args.type} embeddings...")
    if args.type == 'gemini':
        if not args.api_key:
            print("Error: --api_key is required for gemini embeddings.")
            return
        embeddings_list = get_gemini_embeddings(unique_rules, args.api_key, cache_file=args.cache_file)
    else:
        embeddings_list = get_st_embeddings(unique_rules)
    
    lookup = dict(zip(unique_rules, embeddings_list.tolist()))

    print("Saving datasets...")
    suffix = "_Gemini" if args.type == 'gemini' else "_ST"
    for f, df in dataset_dfs.items():
        base_name = os.path.splitext(f)[0]
        df['Embedding'] = df['Natural Language'].map(lambda x: lookup.get(x, None))
        output_path = os.path.join(args.output_dir, f"{base_name}{suffix}.csv")
        df.to_csv(output_path, sep=';', index=False)
        print(f"  Saved to {output_path}")

    print("Done!")

if __name__ == "__main__":
    main()
