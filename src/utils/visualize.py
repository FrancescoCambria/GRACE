import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the results
results_path = 'reports/noisy_st_experiment_results.csv'
if not os.path.exists(results_path):
    print(f"Error: {results_path} not found.")
    exit()

df = pd.read_csv(results_path)

# Filter for requested models
target_models = [
    'DeepCTR DeepFM', 
    'DeepCTR Wide & Deep', 
    'Wide & Deep', 
    'Logistic Regression', 
    'Random Forest'
]
df = df[df['Model'].isin(target_models)]

# Map datasets to numbers
dataset_mapping = {
    'RulesSpotify_Original_100_ST.csv': '1',
    'RulesSpotify_Original_LRT_100_ST.csv': '2',
    'RulesSpotify_Original_270_ST.csv': '3',
    'RulesSpotify_Original_LRT_270_ST.csv': '4',
    'RulesSpotify_Original_1000_ST.csv': '5',
    'RulesSpotify_Original_LRT_1000_ST.csv': '6'
}

df['Dataset_Label'] = df['Dataset'].map(dataset_mapping)

# Drop rows that don't match our mapping (if any)
df = df.dropna(subset=['Dataset_Label'])
df = df.sort_values('Dataset_Label')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
os.makedirs('reports/plots', exist_ok=True)

# We'll pick one training percentage for clarity in the plot, e.g., 80%
# Or we can average across all percentages. Let's use 80% as it's the most complete training.
train_pct = 80
df_plot = df[df['Train %'] == train_pct]

sns.set_theme(style="whitegrid")

for metric in metrics:
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=df_plot, 
        x='Dataset_Label', 
        y=metric, 
        hue='Model',
        palette='viridis'
    )
    
    # Add labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, rotation=90)
    
    plt.title(f'{metric} by Dataset and Model (Train %: {train_pct})', fontsize=15)
    plt.xlabel('Dataset (1:Orig100, 2:LRT100, 3:Orig270, 4:LRT270, 5:Orig1000, 6:LRT1000)', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.ylim(0, 1.1) # Increased limit slightly to accommodate labels
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    output_file = f'reports/plots/{metric.lower().replace("-", "_")}_comparison.png'
    plt.savefig(output_file)
    print(f"Saved plot: {output_file}")
    plt.close()

print("\nVisualization complete. Plots are in reports/plots/")
