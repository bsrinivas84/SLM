"""Demonstrate simplified self-attention computations."""

import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your    (x^1)
     [0.55, 0.87, 0.66],  # journey (x^2)
     [0.57, 0.85, 0.64],  # starts  (x^3)
     [0.22, 0.58, 0.33],  # with    (x^4)
     [0.77, 0.25, 0.10],  # one     (x^5)
     [0.05, 0.80, 0.55]]  # step    (x^6)
)

print("Inputs size:\n", inputs.size())
input_query = inputs[1]   # journey (x^2)

print("Query size:\n", input_query.size())

# attention_scores = torch.matmul(inputs, input_query)
# attention_scores_dot = torch.dot(inputs, input_query)
# print("Attention scores:\n", attention_scores)

#1.Calculating attention scores for the single query vector Journey
attention_scores_New = torch.empty(inputs.size(0))
for i,x_i in enumerate(inputs): 
    attention_scores_New[i] = torch.dot(x_i, input_query)

print("Attention scores (New):\n", attention_scores_New)

#2. Normalize the attention scores
attention_scores_Normalized = torch.nn.functional.softmax(attention_scores_New, dim=0)
print("Normalized Attention scores (New):\n", attention_scores_Normalized)
print("Sum of Normalized Attention scores (New):\n", attention_scores_Normalized.sum())

#3. Compute the context vector
context_vector = torch.zeros(input_query.size(1))
for i,x_i in enumerate(inputs):
    context_vector += attention_scores_Normalized[i] * x_i

print("Context vector:\n", context_vector)
