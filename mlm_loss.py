import torch
import torch.nn as nn
import torch.optim as optim
import random


vocab = ["[PAD]", "[MASK]", "[UNK]", "the", "cat", "sat", "on", "mat", "dog"]
token2id = {tok: i for i, tok in enumerate(vocab)}
id2token = {i: tok for tok, i in token2id.items()}

sentences = [
    "the cat sat on mat",
    "the dog sat on the mat",
    "the cat sat",
    "dog on mat",
]


def encode(sentence):
    return [token2id.get(tok, token2id["[UNK]"]) for tok in sentence.split()]


encoded_dataset = [encode(s) for s in sentences]


def mask_tokens(inputs, mask_token_id=token2id["[MASK]"], mlm_prob=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_prob)
    mask_positions = torch.bernoulli(probability_matrix).bool()
    mask_positions[inputs == token2id["[PAD]"]] = False
    masked_inputs = inputs.clone()
    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            if mask_positions[i, j]:
                prob = random.random()
                if prob < 0.8:
                    masked_inputs[i, j] = mask_token_id
                elif prob < 0.9:
                    masked_inputs[i, j] = random.randint(0, len(vocab) - 1)
            else:
                labels[i, j] = -100
    return masked_inputs, labels


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.transpose(0, 1)
        hidden = self.transformer(emb)
        hidden = hidden.transpose(0, 1)
        logits = self.fc(hidden)
        return logits


def collate_batch(batch, pad_id=token2id["[PAD]"]):
    max_len = max(len(x) for x in batch)
    padded = [x + [pad_id] * (max_len - len(x)) for x in batch]
    return torch.tensor(padded)


batch_size = 2
data_batches = [
    collate_batch(encoded_dataset[i : i + batch_size])
    for i in range(0, len(encoded_dataset), batch_size)
]


model = TinyTransformer(len(vocab))
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(5):
    total_loss = 0
    for batch in data_batches:
        optimizer.zero_grad()
        masked_inputs, labels = mask_tokens(batch)
        logits = model(masked_inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_batches):.4f}")

test_sentence = collate_batch([encode("the cat sat on mat")])
masked, labels = mask_tokens(test_sentence)

print("\nMasked input:", [id2token[i.item()] for i in masked[0]])
logits = model(masked)
pred_ids = torch.argmax(logits, dim=-1)
print("Predictions :", [id2token[i.item()] for i in pred_ids[0]])
print("Labels: ", [id2token[i.item()] if i != -100 else "_" for i in labels[0]])
