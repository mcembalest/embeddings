import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('nanobeir.csv')

plt.style.use('seaborn-v0_8-darkgrid')

fig, axes = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('NanoBEIR Performance Metrics Across Datasets', fontsize=16, y=0.95)

datasets = ['NanoSciFact', 'NanoDBPedia', 'NanoQuoraRetrieval']

models = df['model_name'].unique()
colors = plt.cm.Dark2(np.linspace(0, 1, len(models)))
metrics = ['accuracy', 'precision', 'recall', 'ndcg', 'mrr', 'map']
cutoffs = ['@1', '@3', '@5', '@10']
x_points = [int(x[1:]) for x in cutoffs]
n_metrics = len(metrics)
n_datasets = len(datasets)
fig, axes = plt.subplots(n_metrics, n_datasets, figsize=(20, 25))
fig.suptitle('NanoBEIR Performance Metrics Across Datasets', fontsize=36, y=1.02)

for metric_idx, metric in enumerate(metrics):
    for dataset_idx, dataset in enumerate(datasets):
        ax = axes[metric_idx, dataset_idx]
        
        # Different handling based on metric type
        for model_idx, model in enumerate(models):
            model_df = df[df['model_name'] == model]
            
            if metric in ['accuracy', 'precision', 'recall']:
                # Line plot for metrics with multiple cutoffs
                values = []
                for cutoff in cutoffs:
                    metric_name = f'{dataset}_cosine_{metric}{cutoff}'
                    if metric_name in df.columns:
                        values.append(model_df[metric_name].iloc[0])
                    else:
                        values.append(0)
                
                ax.plot(x_points, values, marker='o', color=colors[model_idx], 
                       label=model if dataset_idx == 0 else "")
                ax.set_xticks(x_points)
                ax.set_xticklabels(cutoffs)
                ax.set_xlabel('k')
            
            else:
                # Bar plot for single-value metrics
                if metric == 'ndcg':
                    cutoff = '@10'
                elif metric == 'mrr':
                    cutoff = '@10'
                else:  # map
                    cutoff = '@100'
                
                metric_name = f'{dataset}_cosine_{metric}{cutoff}'
                if metric_name in df.columns:
                    value = model_df[metric_name].iloc[0]
                    # Calculate bar position
                    bar_width = 0.8 / len(models)
                    bar_pos = model_idx * bar_width - (len(models) * bar_width / 2) + bar_width/2
                    ax.bar(bar_pos, value, bar_width, color=colors[model_idx],
                          label=model if dataset_idx == 0 else "")
                
                ax.set_xticks([0])
                ax.set_xticklabels([cutoff])
                ax.set_xlabel(metric.upper())
        
        # Common plot settings
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        if dataset_idx == 0:
            ax.set_ylabel(f'{metric.upper()} Score')
        if metric_idx == 0:
            ax.set_title(dataset)

# Add legend and adjust layout
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.08, 0.5))
plt.tight_layout()
plt.subplots_adjust(right=0.9)
plt.savefig('plots/nanobeir_metrics.png', dpi=300, bbox_inches='tight')
plt.close()