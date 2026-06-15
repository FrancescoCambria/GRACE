import os
import pandas as pd
import numpy as np
import argparse

def add_noise(df, noise_level=0.15):
    df_noisy = df.copy()
    noise_mask = np.random.choice([True, False], size=len(df_noisy), p=[noise_level, 1-noise_level])
    df_noisy.loc[noise_mask, 'tag'] = 1 - df_noisy.loc[noise_mask, 'tag'].astype(int)
    return df_noisy

def main():
    parser = argparse.ArgumentParser(description="Generate noisy datasets from tagged datasets.")
    parser.add_argument("--input_dir", default="RulesSpotify/TaggedDatasets", help="Directory containing tagged datasets.")
    parser.add_argument("--output_dir", default="RulesSpotify/TaggedDatasets_Noisy", help="Directory to save noisy datasets.")
    parser.add_argument("--noise_level", type=float, default=0.15, help="Level of noise to add (default: 0.15).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(args.input_dir) if f.endswith(".csv")]
    for f in files:
        print(f"Adding {args.noise_level} noise to {f}...")
        df = pd.read_csv(os.path.join(args.input_dir, f), sep=';')
        df_noisy = add_noise(df, args.noise_level)
        df_noisy.to_csv(os.path.join(args.output_dir, f), sep=';', index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
