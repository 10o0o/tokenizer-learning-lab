import torch

# One sequence of four token embeddings, with three values per token.
sample = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.5],
        [-1.0, 0.0, 1.0],
    ]
)

scores = sample @ sample.T
weights = torch.softmax(scores, dim=-1)
context_vectors = weights @ sample


W_q, W_k, W_v = torch.rand(3, 2), torch.rand(3, 2), torch.rand(3, 2)
Q = sample @ W_q
K = sample @ W_k
V = sample @ W_v

score = Q @ K.T
mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
masked_score = score.masked_fill(mask, float("-inf"))
weights = torch.softmax(masked_score / K.shape[-1] ** 0.5, dim=-1)
context_vector = weights @ V


class SelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.W_q, self.W_k, self.W_v = (
            torch.rand(d_in, d_out),
            torch.rand(d_in, d_out),
            torch.rand(d_in, d_out),
        )
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # scores = Q @ K.T
        scores = Q @ K.transpose(-2, -1)
        mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        masked_score = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(masked_score / K.shape[-1] ** 0.5, dim=-1)
        dropped_weights = self.dropout(weights)
        context_vector = dropped_weights @ V
        return context_vector


layer = SelfAttention(3, 2)
layer2 = SelfAttention(3, 2)

batch = torch.stack((sample, sample), dim=0)

out1 = layer(batch)
out2 = layer2(batch)

combined = torch.cat((out1, out2), dim=-1)
print(combined.shape)

W_q_all = torch.cat((layer.W_q, layer2.W_q), dim=-1)
Q_all = batch @ W_q_all
Q_split = Q_all.reshape(2, 4, 2, 2)
Q_heads = Q_split.transpose(1, 2)

W_k_all = torch.cat((layer.W_k, layer2.W_k), dim=-1)
K_all = batch @ W_k_all
K_heads = K_all.reshape(2, 4, 2, 2).transpose(1, 2)

scores_by_head = Q_heads @ K_heads.transpose(-2, -1)

W_v_all = torch.cat((layer.W_v, layer2.W_v), dim=-1)
V_all = batch @ W_v_all
V_heads = V_all.reshape(2, 4, 2, 2).transpose(1, 2)

mask = torch.triu(torch.ones_like(scores_by_head, dtype=torch.bool), diagonal=1)
masked_scores_by_head = scores_by_head.masked_fill(mask, float("-inf"))
weights_by_head = torch.softmax(
    masked_scores_by_head / K_heads.shape[-1] ** 0.5, dim=-1
)
context_by_head = weights_by_head @ V_heads
context_by_token = context_by_head.transpose(1, 2)
combined_batched = context_by_token.reshape(2, 4, 4)


layer.eval()
layer2.eval()
reference = torch.cat((layer(batch), layer2(batch)), dim=-1)

print(combined_batched.shape)
print(torch.allclose(reference, combined_batched))
