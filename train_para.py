import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

json_data = "data_2.json"
with open(json_data, "r", encoding="utf-8") as f:
    data = json.load(f)

pairs = []
for para in data:
    for question in para['questions']:
        pairs.append({'question': question, 'answer': para['paragraph_text']})

print(len(pairs))
for i in range(10):
    print(pairs[i])

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