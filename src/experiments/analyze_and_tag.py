import pandas as pd
import argparse
import os
import sys
import re
import csv

def evaluate_all_criteria(row):
    """
    Evaluates all criteria from across various legacy scripts.
    Returns a dictionary of criteria (0 or 1).
    """
    # 1. Data Extraction & Normalization
    body_raw = str(row.get('Body', ''))
    head_raw = str(row.get('Head', ''))
    body_names = str(row.get('Body Node Names', '')).lower()
    head_names = str(row.get('Head Node Names', '')).lower()
    anchor = str(row.get('Anchor Label', '')).lower()
    
    body_struct = body_raw.lower()
    head_struct = head_raw.lower()
    all_struct = (body_struct + " " + head_struct + " " + anchor).lower()
    all_names = (body_names + " " + head_names).lower()
    all_raw = body_raw + ", " + head_raw
    
    body_ids = [idx.strip() for idx in str(row.get('Body Node IDs', '')).split(',') if idx.strip()]
    head_ids = [idx.strip() for idx in str(row.get('Head Node IDs', '')).split(',') if idx.strip()]
    body_p = [p.strip() for p in body_raw.split(', ') if p.strip()]
    head_p = [p.strip() for p in head_raw.split(', ') if p.strip()]
    
    try:
        conf = float(row.get('Confidence', 0))
    except (ValueError, TypeError):
        conf = 0.0
        
    try:
        supp = float(row.get('Support', 0))
    except (ValueError, TypeError):
        supp = 0.0

    # 2. Pre-computations
    entity_types = set(re.findall(r'\(([a-zA-Z0-9_]+)\)', all_struct))
    # Add common types if they exist in text but weren't caught by regex
    for t in ['user', 'playlist', 'artist', 'genre', 'track', 'song']:
        if t in all_struct:
            entity_types.add(t)
    
    type_count = len(entity_types)
    pattern_count_h = len(head_p)
    pattern_count_b = len(body_p)
    total_patterns = pattern_count_b + pattern_count_h
    total_edges = all_struct.count('->') + all_struct.count('<-')
    avg_complexity = total_edges / total_patterns if total_patterns > 0 else 0
    
    # Structural features
    has_convergence = len(set(head_ids)) < pattern_count_h if pattern_count_h > 1 else False
    
    is_complex = type_count >= 3
    is_bridge = "playlist" in body_struct and ("artist" in head_struct or "song" in head_struct)
    is_expert = "user" in body_struct and "genre" in head_struct

    # Shortcut detection
    body_edge_counts = [p.count('->') + p.count('<-') for p in body_p]
    head_edge_counts = [p.count('->') + p.count('<-') for p in head_p]
    is_shortcut = False
    if body_p and head_p:
        max_b = max(body_edge_counts) if body_edge_counts else 0
        min_h = min(head_edge_counts) if head_edge_counts else 0
        if max_b > min_h and any(idx in body_ids for idx in head_ids):
            is_shortcut = True

    # Textual macro groups
    macro_groups = {
        'rock_metal': ['rock', 'punk', 'metal', 'grunge', 'hardcore', 'emo', 'alternative', 'indie'],
        'urban_dance': ['pop', 'hip hop', 'rap', 'trap', 'r&b', 'soul', 'dance', 'house', 'techno', 'edm', 'beats', 'drill'],
        'traditional_roots': ['jazz', 'classical', 'blues', 'folk', 'standards', 'orchestral', 'country', 'bluegrass', 'americana'],
        'latin_global': ['mexicano', 'regional', 'banda', 'corridos', 'español', 'latin', 'brazilian', 'k-pop', 'afrobeat', 'reggaeton'],
        'ambient_wellness': ['meditation', 'healing', 'frequencies', 'sleep', 'solfeggio', 'relaxation', 'calm', 'mindfulness', 'yoga', 'zen', 'ambient', 'new age'],
        'activity_lifestyle': ['gym', 'workout', 'fitness', 'running', 'hiking', 'camping', 'study', 'reading', 'focus', 'chill', 'lofi', 'cocktail', 'dinner', 'party', 'driving', 'road trip']
    }
    
    macro_hits = {k: any(g in all_names for g in v) for k, v in macro_groups.items()}
    macro_hit_counts = {k: sum(1 for g in v if g in all_names) for k, v in macro_groups.items()}

    # Motif counting helper
    def count_motif(pattern, text):
        clean_text = text.lower().replace(' ', '')
        clean_pattern = pattern.lower().replace(' ', '')
        return clean_text.count(clean_pattern)

    # 3. Define Criteria
    criteria = {
        # --- Basic Structural ---
        'Struct_Complex': 1 if is_complex else 0,
        'Struct_Bridge': 1 if is_bridge else 0,
        'Struct_Expert': 1 if is_expert else 0,
        'Struct_Convergent_Head': 1 if has_convergence else 0,
        'Struct_Shortcut': 1 if is_shortcut else 0,
        'Struct_High_Complexity': 1 if avg_complexity > 2.5 else 0,
        
        # --- Macro Groups (Spotify) ---
        'Macro_RockMetal': 1 if macro_hits['rock_metal'] else 0,
        'Macro_UrbanDance': 1 if macro_hits['urban_dance'] else 0,
        'Macro_TraditionalRoots': 1 if macro_hits['traditional_roots'] else 0,
        'Macro_LatinGlobal': 1 if macro_hits['latin_global'] else 0,
        'Macro_AmbientWellness': 1 if macro_hits['ambient_wellness'] else 0,
        'Macro_ActivityLifestyle': 1 if macro_hits['activity_lifestyle'] else 0,
        
        # --- Combined Macro/Structural (from genre-specific script) ---
        'Crit_Global': 1 if macro_hits['latin_global'] else 0,
        'Crit_Heavy_Expert': 1 if (macro_hits['rock_metal'] and is_expert) else 0,
        'Crit_Urban_Complex': 1 if (macro_hits['urban_dance'] and is_complex) else 0,
        'Crit_Traditional_Bridge': 1 if (macro_hits['traditional_roots'] and is_bridge) else 0,
        'Crit_Wellness_Expert': 1 if (macro_hits['ambient_wellness'] and is_expert) else 0,
        'Crit_Lifestyle_Complex': 1 if (macro_hits['activity_lifestyle'] and is_complex) else 0,
        
        # --- Motifs (Spotify) ---
        'Motif_Song_User_Playlist': 1 if count_motif('(Song)-[IN]->(Playlist)-[CREATED_BY]->(User)', all_raw) > 0 else 0,
        'Motif_Song_Genre_Playlist': 1 if count_motif('(Song)-[IN]->(Playlist)-[OF]->(Genre)', all_raw) > 0 else 0,
        'Motif_Artist_Genre': 1 if count_motif('(Artist)-[OF]->(Genre)', all_raw) > 0 else 0,
        'Motif_Artist_Song_Genre_Playlist': 1 if count_motif('(Artist)-[SING]->(Song)-[IN]->(Playlist)-[OF]->(Genre)', all_raw) > 0 else 0,
        'Motif_Artist_Song_Playlist': 1 if count_motif('(Artist)-[SING]->(Song)-[IN]->(Playlist)', all_raw) > 0 else 0,
        'Motif_Artist_Song_User_Playlist': 1 if count_motif('(Artist)-[SING]->(Song)-[IN]->(Playlist)-[CREATED_BY]->(User)', all_raw) > 0 else 0,
        'Motif_Playlist_Genre': 1 if count_motif('(Playlist)-[OF]->(Genre)', all_raw) > 0 else 0,
        'Motif_User_Playlist': 1 if count_motif('(Playlist)-[CREATED_BY]->(User)', all_raw) > 0 else 0,

        'Motif_Artist_Genre_3plus': 1 if count_motif('(Artist)-[OF]->(Genre)', all_raw) >= 3 else 0,
        'Motif_User_Playlist_2plus': 1 if count_motif('(Playlist)-[CREATED_BY]->(User)', all_raw) >= 2 else 0,
        'Motif_Song_Genre_Playlist_2plus': 1 if count_motif('(Song)-[IN]->(Playlist)-[OF]->(Genre)', all_raw) >= 2 else 0,
        'Motif_Artist_Song_Playlist_2plus': 1 if count_motif('(Artist)-[SING]->(Song)-[IN]->(Playlist)', all_raw) >= 2 else 0,
        
        # --- Original Legacy Logic ---
        'Orig_Trivial_Artifact': 1 if ("created_by" in head_struct and "user" in head_struct and "playlist" in head_struct and conf > 0.85 and "genre" not in all_struct and "artist" not in all_struct) else 0,
        'Orig_Genre_Crossover': 1 if ("genre" in body_struct and "genre" in head_struct and not set(body_names.split(',')).intersection(set(head_names.split(','))) and conf > 0.8) else 0,
        'Orig_Human_Curator': 1 if ("playlist" in body_struct and "user" in head_struct and "spotify" not in all_names and (" " in body_names or len(body_names) > 20) and conf > 0.85) else 0,
        'Orig_Artist_MicroTrend': 1 if (("artist" in body_struct or "artist" in head_struct) and supp < 0.005 and conf > 0.8) else 0,
        'Orig_Vibe_to_Genre': 1 if ("playlist" in body_struct and "genre" in head_struct and " " in body_names and conf > 0.8) else 0,

        # --- Domain: Law ---
        'Law_Cites': 1 if "cites" in all_struct else 0,
        'Law_Dept': 1 if "department" in head_struct else 0,
    }
    
    # Special "Basic4" logic
    criteria['Basic4'] = 1 if (
        criteria['Macro_RockMetal'] or 
        criteria['Motif_Artist_Genre_3plus'] or 
        criteria['Macro_UrbanDance'] or 
        criteria['Motif_User_Playlist_2plus']
    ) else 0
    
    return criteria

def main():
    parser = argparse.ArgumentParser(description="Unified script for analyzing and tagging rule datasets.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", help="Output CSV file. If not provided, creates a copy with '_tagged' suffix.")
    parser.add_argument("--sep", default=";", help="CSV separator (default: ';').")
    parser.add_argument("--criteria", nargs='*', help="Criteria to use for final 'tag'. If multiple, tag=1 if ANY match.")
    parser.add_argument("--analyze", action="store_true", help="Perform analysis (individual distribution and co-occurrence).")
    parser.add_argument("--search", nargs='*', help="Search terms to create dynamic criteria columns.")
    parser.add_argument("--all-criteria", action="store_true", help="Add ALL built-in criteria columns to the output.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found {args.input}")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(args.input, sep=args.sep)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # Try with comma if semicolon fails and was default
        if args.sep == ";":
             try:
                 df = pd.read_csv(args.input, sep=",")
                 args.sep = ","
                 print(f"Detected comma separator instead.")
             except:
                 return
        else:
            return

    # 2. Evaluate Built-in Criteria
    print("Evaluating built-in criteria...")
    criteria_results = []
    for _, row in df.iterrows():
        criteria_results.append(evaluate_all_criteria(row))
    
    crit_df = pd.DataFrame(criteria_results)
    
    # 3. Evaluate Search Criteria
    search_cols = []
    if args.search:
        print(f"Evaluating search terms: {args.search}")
        text_cols = ['Body', 'Head', 'Body Node Names', 'Head Node Names', 'Anchor Label']
        # Only use columns that exist in the dataframe
        available_text_cols = [c for c in text_cols if c in df.columns]
        if available_text_cols:
            text_data = df[available_text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
            for term in args.search:
                col_name = f"Search_{term}"
                df[col_name] = text_data.str.contains(term.lower()).astype(int)
                search_cols.append(col_name)
        else:
            print("Warning: None of the search columns (Body, Head, etc.) found in input.")

    # 4. Merge criteria into main DF
    if args.all_criteria or args.criteria:
        # Determine which criteria columns to actually add
        cols_to_add = []
        if args.all_criteria:
            cols_to_add = list(crit_df.columns)
        elif args.criteria:
            cols_to_add = [c for c in args.criteria if c in crit_df.columns]
            
        for col in cols_to_add:
            df[col] = crit_df[col].values

    # 5. Determine Tag
    if args.criteria:
        print(f"Determining 'tag' based on: {args.criteria}")
        tag_mask = pd.Series([False] * len(df))
        applied_logics = []
        
        for crit in args.criteria:
            if crit in crit_df.columns:
                tag_mask |= (crit_df[crit] == 1)
            elif f"Search_{crit}" in df.columns:
                tag_mask |= (df[f"Search_{crit}"] == 1)
            else:
                print(f"Warning: Criterion '{crit}' not found.")
        
        df['tag'] = tag_mask.astype(int)
        
        # Update Tag_Logic
        def get_logic(row_idx):
            active = []
            for crit in args.criteria:
                if crit in crit_df.columns and crit_df.iloc[row_idx][crit] == 1:
                    active.append(crit)
                elif f"Search_{crit}" in df.columns and df.iloc[row_idx][f"Search_{crit}"] == 1:
                    active.append(f"Search_{crit}")
            return " + ".join(active) if active else "Standard"
        
        df['Tag_Logic'] = [get_logic(i) for i in range(len(df))]

    # 6. Perform Analysis if requested
    if args.analyze:
        print("\n--- Analysis Report ---")
        analysis_cols = list(crit_df.columns) + search_cols
        analysis_cols = [c for c in analysis_cols if c in df.columns or c in crit_df.columns]
        
        # Temporary merge for analysis if not already merged
        temp_df = pd.concat([df, crit_df[[c for c in crit_df.columns if c not in df.columns]]], axis=1)
        
        counts = temp_df[analysis_cols].sum()
        freqs = (counts / len(df) * 100).round(2)
        summary = pd.DataFrame({'Count': counts, 'Frequency (%)': freqs}).sort_values(by='Count', ascending=False)
        print("\nCriteria Distribution:")
        print(summary.to_string())
        
        print("\nCo-occurrence Matrix (Rules matching both):")
        matrix = pd.DataFrame(index=analysis_cols, columns=analysis_cols)
        for c1 in analysis_cols:
            for c2 in analysis_cols:
                matrix.loc[c1, c2] = int(((temp_df[c1] == 1) & (temp_df[c2] == 1)).sum())
        print(matrix.loc[summary.index, summary.index].to_string())

    # 7. Save Output
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_tagged{ext}"
    
    df.to_csv(output_path, sep=args.sep, index=False)
    print(f"\nSuccessfully saved to: {output_path}")

if __name__ == "__main__":
    main()
