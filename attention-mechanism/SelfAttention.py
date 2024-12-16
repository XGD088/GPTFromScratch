import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        values = x @ self.W_value
        querys = x @ self.W_query
        attention_scores = querys @ keys.T
        ## prevent Gradient Explosion
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attention_weights @ values
        return context_vec

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key2 = x_2 @ W_key
value2 = x_2 @ W_value

attention_score22 = query_2.dot(key2)

keys = inputs @ W_key
values = inputs @ W_key
print("keys:", keys)
print(query_2)

attention_score2 = query_2 @ keys.T

d_k = keys.shape[-1]
print(d_k)
attn_weights_2 = torch.softmax(attention_score2 / d_k**0.5, dim=-1)

context_vec_2 = attn_weights_2 @ values


torch.manual_seed(789)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))




a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],  # A
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

print("a.trans", a.transpose(2, 3))

print(a @ a.transpose(2, 3))
