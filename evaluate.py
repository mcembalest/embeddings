from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator
import pandas as pd
from datetime import datetime
import os

# Model configuration
model_name = "sentence-transformers/static-retrieval-mrl-en-v1"
# model_name = "test_static_embeddings/checkpoint-33"
# model_name = "sentence-transformers/all-MiniLM-L6-V2"
# model_name = "sentence-transformers/all-mpnet-base-v2
# model_name = "nomic-ai/modernbert-embed-base"
# model_name = "nomic-ai/nomic-embed-text-v1.5"
trust_remote_code = False

model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code).to("cpu")

# Run evaluation
evaluator = NanoBEIREvaluator(
    dataset_names=[
        'scifact', 'dbpedia', 'quoraretrieval'
    ],
    show_progress_bar=True
)
results = evaluator(model)

# Prepare results with additional metadata
results['model_name'] = model_name
results['timestamp'] = datetime.now().isoformat()

# Convert results to DataFrame
new_results_df = pd.DataFrame([results])

# Load existing results if available, otherwise create new DataFrame
csv_path = 'nanobeir.csv'
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    results_df = pd.concat([existing_df, new_results_df], ignore_index=True)
else:
    results_df = new_results_df

# Save updated results
results_df.to_csv(csv_path, index=False)

# Print summary of latest results
print(f"\nResults for model: {model_name}")
print(f"Mean accuracy@1: {results['NanoBEIR_mean_cosine_accuracy@1']:.3f}")
print(f"Mean NDCG@10: {results['NanoBEIR_mean_cosine_ndcg@10']:.3f}")