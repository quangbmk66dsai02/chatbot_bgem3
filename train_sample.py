import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

file_path = "data.json"

with open(file_path, "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)
train_examples = [InputExample(texts=[pair["question"], pair["answer"]]) for pair in qa_pairs]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
# Retrieve the first batch from the DataLoader
first_batch = next(iter(train_dataloader))


model = SentenceTransformer('BAAI/bge-m3')


train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100
)
model.save('fine-tuned-sentence-transformer')
