import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot Spotify results comparison matrix.")
    parser.add_argument(
        "--input_file", 
        default="rules/spotify_for_plots/Results_forGraphs_1000_Joint.xlsx", 
        help="Path to input Excel file."
    )
    parser.add_argument(
        "--output_dir", 
        default="reports/plot", 
        help="Path to output directory."
    )
    parser.add_argument(
        "--exclude", 
        nargs='+', 
        default=[], 
        help="Feature configurations (e.g., 'metrics', 'st', 'rotate', etc.) to exclude."
    )
    parser.add_argument(
        "--include_only", 
        nargs='+', 
        default=None, 
        help="If specified, only plot these configurations (e.g., 'metrics', 'st', etc.)."
    )

    args = parser.parse_args()

    file_path = args.input_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_excel(file_path)

    # Define metrics (columns of the matrix)
    metrics = [
        ('Avg_Accuracy', 'Accuracy'),
        ('Avg_Precision', 'Precision'),
        ('Avg_Recall', 'Recall'),
        ('Avg_F1-Score', 'F1-Score')
    ]

    # Dataset row mapping
    dataset_mapping = {
        'RulesSpotify_Genre_Specific_1000_Balanced.csv': 'Music Expert',
        'RulesSpotify_Structural_1000_Balanced.csv': 'Network Analyst',
        'RulesSpotify_Basic4_1000.csv': 'General User'
    }

    row_datasets = [
        'RulesSpotify_Genre_Specific_1000_Balanced.csv',
        'RulesSpotify_Structural_1000_Balanced.csv',
        'RulesSpotify_Basic4_1000.csv'
    ]

    # Configuration renaming and color mapping
    config_rename = {
        'metrics': 'Statistical Features',
        'rotate': 'Graph Features',
        'st': 'Text Features',
        'metrics+rotate': 'Statistical + Graph Features',
        'metrics+st': 'Statistical + Text Features',
        'rotate+st': 'Graph + Text Features',
        'metrics+rotate+st': 'Statistical + Graph + Text Features'
    }

    # Original color palette
    config_colors = {
        'metrics': '#4F46E5',           # Indigo
        'rotate': "#D4DB6B",            # Cyan
        'st': '#10B981',                # Emerald
        'metrics+rotate': "#4DA768",    # Amber
        'metrics+st': "#E993BE",        # Pink
        'rotate+st': "#5C92F6",        # Violet
        'metrics+rotate+st': "#BD4242"  # Rose/Red
    }

    configs_order = [
        'metrics',
        'rotate',
        'st',
        'metrics+rotate',
        'metrics+st',
        'rotate+st',
        'metrics+rotate+st'
    ]

    # Filter configurations based on command-line arguments
    if args.include_only:
        configs_order = [c for c in configs_order if c in args.include_only]
    if args.exclude:
        configs_order = [c for c in configs_order if c not in args.exclude]

    if not configs_order:
        print("Error: No configurations left to plot after filtering.")
        return

    # Tighter layout figsize (18x11)
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)

    # Get unique sorted test sizes
    test_sizes = sorted(df['test_size'].unique())
    
    # Format labels to 2 decimal places, except 0.925 which is 3
    x_labels = [f"{val:.2f}" if val != 0.925 else "0.925" for val in test_sizes]

    # We will iterate through each row (dataset) and column (metric)
    for row_idx, ds_name in enumerate(row_datasets):
        ds_label = dataset_mapping[ds_name]
        df_ds = df[df['Dataset'] == ds_name]

        for col_idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            # Plot each configuration in order
            for config_val in configs_order:
                df_config = df_ds[df_ds['Include'] == config_val].sort_values('test_size')

                if not df_config.empty:
                    label = config_rename.get(config_val, config_val)
                    color = config_colors.get(config_val)
                    
                    # Plot linearly over the actual test_size values
                    ax.plot(
                        df_config['test_size'],
                        df_config[metric_col],
                        marker='o',
                        linewidth=2,
                        markersize=5,
                        color=color,
                        label=label
                    )

            # Column headers on the top row
            if row_idx == 0:
                ax.set_title(metric_label, fontsize=16, fontweight='bold', pad=12)

            # Row labels on the left column
            if col_idx == 0:
                ax.set_ylabel(ds_label, fontsize=14, fontweight='bold')

            # X-axis labels and tick configuration
            ax.set_ylim(0.4, 1)
            ax.set_xticks(test_sizes)
            # Rotate labels 45 degrees and align right to prevent overlap
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            
            # X-axis title only on the bottom row
            if row_idx == 2:
                ax.set_xlabel("Test Size", fontsize=12)

            # Style the spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.grid(True, linestyle='--', alpha=0.3)

    # Get handles and labels for the legend from the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Place a single legend below the subplots, shifted slightly above (y=0.04)
    # Since only 4 configurations will be shown, they will occupy a single line.
    fig.legend(
        handles, 
        labels, 
        loc='lower center', 
        ncol=min(4, len(configs_order)), 
        bbox_to_anchor=(0.5, 0.04), 
        fontsize=12,
        frameon=True,
        facecolor='white',
        edgecolor='lightgray'
    )

    # Adjusted layout rect: starting bottom at 0.1 to clear the shifted-up legend
    plt.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3, rect=[0.01, 0.1, 0.99, 0.98])

    save_path = os.path.join(output_dir, 'spotify_results_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison matrix plot to: {save_path}")

if __name__ == "__main__":
    main()