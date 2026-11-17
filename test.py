import torch

ids = [97, 98, 99, 100, 101, 102, 103, 104]
context_size = 4
stride = 2

x_batch = []
y_batch = []

for i in range(0, len(ids) - context_size, 2):
    x = []
    y = []

    for j in range(context_size):
        x.append(ids[i + j])
        y.append(ids[i + j + 1])

    x_batch.append(x)
    y_batch.append(y)

print(x_batch)


x_tensor = torch.tensor(x_batch, dtype=torch.long)
embedding = torch.nn.Embedding(num_embeddings=256, embedding_dim=3)

x_emb = embedding(x_tensor)
# print(x_emb.shape)  # torch.Size([2, 4, 3])
# print(torch.equal(x_emb[0, 3], x_emb[1, 1]))  # True

positions = torch.arange(context_size)
table_emb = torch.nn.Embedding(num_embeddings=4, embedding_dim=3)
pos_table = table_emb(positions)

combined = x_emb + pos_table


print(f"combined.shape: {combined.shape}")
print(torch.equal(combined[0, 3], combined[1, 1]))


sample = combined[0]
query = sample[1]

scores = sample @ query
print(scores)
print(torch.equal(scores[0], torch.dot(sample[0], query)))

weights = torch.softmax(scores, dim=-1)

attribute = weights[0] * sample[0]
context_vector = attribute
context_vector = context_vector + weights[1] * sample[1]
