import torch

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
)

# calculate single context vector
# query is also called "key"
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, inputs_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(inputs_i, query)

attn_weights_2_naive = softmax_naive(attn_scores_2)

context_vec_2 = torch.zeros(query.shape)

for i, inputs_i in enumerate(inputs):
    context_vec_2 += inputs[i] * attn_weights_2_naive[i]

print(attn_weights_2_naive)
print(attn_weights_2_naive.sum())
print(context_vec_2)


#caculate all context vector
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=1)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=1))



all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C