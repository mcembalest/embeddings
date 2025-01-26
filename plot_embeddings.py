import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import os
from sentence_transformers import SentenceTransformer
import umap
from tqdm import tqdm

df = pd.read_csv('nanobeir.csv')
models = df['model_name'].unique()

def get_embeddings(model, texts, model_name, dataset_name, batch_size=32):
    os.makedirs('saved_embeddings', exist_ok=True)
    cache_file = f'saved_embeddings/embeddings_{dataset_name}_{model_name.replace("/", "_")}.npy'
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)
    
    print(f"Generating new embeddings for {model_name}")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True).cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    np.save(cache_file, embeddings)
    return embeddings

datasets = {
    'NanoSciFact': 'zeta-alpha-ai/NanoSciFact',
    'NanoQuoraRetrieval': 'zeta-alpha-ai/NanoQuoraRetrieval',
    'NanoDBPedia': 'zeta-alpha-ai/NanoDBPedia'
}

umap_reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

plt.style.use('seaborn-v0_8-darkgrid')
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

for dataset_name, dataset_path in datasets.items():
    print(f"\nProcessing {dataset_name}...")
    
    dataset = load_dataset(dataset_path, 'corpus', split='train')
    texts = dataset['text']

    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_idx, model_name in enumerate(models):
        print(f"Processing model: {model_name}")
        model = SentenceTransformer(model_name).to("cpu")
        embeddings = get_embeddings(model, texts, model_name, dataset_name)
        reduced_embeddings = umap_reducer.fit_transform(embeddings)
        
        ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=[colors[model_idx]],
            label=model_name,
            alpha=0.6,
            s=50
        )
    
    ax.set_title(f'UMAP Embeddings - {dataset_name} (Sample Size: {len(texts)})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'plots/umap_{dataset_name}_sample.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nDone! Check the generated PNG files.")