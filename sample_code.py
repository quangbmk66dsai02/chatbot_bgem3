from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,

)
from sentence_transformers.losses import MultipleNegativesRankingLoss, TripletLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator, RerankingEvaluator, InformationRetrievalEvaluator

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer("BAAI/bge-m3")

from datasets import load_dataset

# 3. Load a dataset to finetune on
from datasets import load_dataset

dataset = load_dataset("meandyou200175/data_split_csv")
dataset = dataset.rename_column("pos", "positive")
dataset = dataset.rename_column("neg", "negative")

def flatten_columns(example):
    example['pos'] = example['pos'][0] if isinstance(example['pos'], list) else example['pos']
    example['neg'] = example['neg'][0] if isinstance(example['neg'], list) else example['neg']
    return example

# 3. Áp dụng hàm trên toàn bộ tập dữ liệu
eval_dataset = dataset["eval"]
# dataset = dataset.map(flatten_columns)
train_dataset = dataset["train"]

test_dataset = dataset["test"]
# loss = MultipleNegativesRankingLoss(model) OnlineContrastiveLoss
from sentence_transformers.losses import MultipleNegativesRankingLoss, TripletLoss, OnlineContrastiveLoss

loss = MultipleNegativesRankingLoss(model)

corpus = {}
queries = {}
relevant_docs = {}

for idx, data in enumerate(eval_dataset):
    query_id = f"{2*idx}"
    positive_id = f"{2*idx+1}"

    
    # Add to corpus
    corpus[positive_id] = data['positive']

    
    # Add to queries
    queries[query_id] = data['query']
    
    # Map relevant docs
    relevant_docs[query_id] = {positive_id}

    ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    write_csv=True,
    show_progress_bar=True,
    accuracy_at_k = [1, 2, 5, 10, 100],
    precision_recall_at_k = [1, 2, 5, 10, 100],
    mrr_at_k = [1, 2, 5, 10, 100],
    # name="BeIR-quora-dev",
)
result = ir_evaluator(model)
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=ir_evaluator,
)
trainer.train()