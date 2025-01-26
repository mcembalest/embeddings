from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.losses import MultipleNegativesRankingLoss
from transformers import AutoTokenizer, TrainerCallback
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import json

with open('train_examples.json', 'r') as f:
    train_examples = json.load(f)
dataset = Dataset.from_list(train_examples)

# Create train/validation split
dataset_splits = dataset.train_test_split(test_size=0.2)  # Using 20% for evaluation
train_dataset = dataset_splits["train"]
eval_dataset = dataset_splits["test"]

# Use the existing text pairs and their labels for evaluation
eval_sentences1 = []
eval_sentences2 = []
eval_scores = []

for example in eval_dataset:
    eval_sentences1.append(example['text'])
    eval_sentences2.append(example['text_pair'])
    eval_scores.append(example['label'])  # Using the actual similarity scores from your data



tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
static_embedding = StaticEmbedding(tokenizer, embedding_dim=1024)
model = SentenceTransformer(modules=[static_embedding]).to("cpu")

args = SentenceTransformerTrainingArguments(
    output_dir="test_static_embeddings",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Smaller batch size for CPU
    learning_rate=2e-1,
    warmup_ratio=0.1,
    logging_steps=10,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    # Disable features that aren't needed
    fp16=False,
    bf16=False,
)

# Custom callback for CSV logging
class CSVLoggingCallback(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.metrics = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Add step number to metrics
            logs['step'] = state.global_step
            self.metrics.append(logs)
            
            # Save to CSV
            pd.DataFrame(self.metrics).to_csv(self.csv_path, index=False)
    
    def on_train_end(self, args, state, control, **kwargs):
        # Create visualization
        metrics_df = pd.DataFrame(self.metrics)
        
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['step'], metrics_df['loss'], label='Training Loss')
        if 'eval_loss' in metrics_df.columns:
            plt.plot(metrics_df['step'], metrics_df['eval_loss'], label='Eval Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.savefig('training_progress.png')
        plt.close()

loss = MultipleNegativesRankingLoss(model)

evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_sentences1,
    sentences2=eval_sentences2,
    scores=eval_scores,
    batch_size=32
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=evaluator,
    callbacks=[CSVLoggingCallback("local_metrics.csv")]
)

trainer.train()