"""Compute self-attention for all input tokens."""

import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your    (x^1)
     [0.55, 0.87, 0.66],  # journey (x^2)
     [0.57, 0.85, 0.64],  # starts  (x^3)
     [0.22, 0.58, 0.33],  # with    (x^4)
     [0.77, 0.25, 0.10],  # one     (x^5)
     [0.05, 0.80, 0.55]]  # step    (x^6)
)


#1.Calculating attention scores for the single query vector Journey
attention_scores_New = torch.empty(6,6)
# for i,x_i in enumerate(inputs):
# 	for j,x_j in enumerate(inputs):
# 		attention_scores_New[i,j] = torch.dot(x_i, x_j)

# attention_scores_New = torch.matmul(inputs, inputs.T)
# print("Attention scores (New):\n", attention_scores_New)

attention_scores_New = inputs @ inputs.T
print("Attention scores (New):\n", attention_scores_New)

#2 Normalize the attention scores
attention_scores_Normalized = torch.nn.functional.softmax(attention_scores_New, dim=1)  
print("Normalized Attention scores (New):\n", attention_scores_Normalized)

#3. Compute the context vector
context_vectors = attention_scores_Normalized @ inputs
print("Context vectors:\n", context_vectors)
