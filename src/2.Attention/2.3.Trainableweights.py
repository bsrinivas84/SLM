import torch

inputs = torch.tensor(
	[[0.43, 0.15, 0.89],  # Your    (x^1)
	 [0.55, 0.87, 0.66],  # journey (x^2)
	 [0.57, 0.85, 0.64],  # starts  (x^3)
	 [0.22, 0.58, 0.33],  # with    (x^4)
	 [0.77, 0.25, 0.10],  # one     (x^5)
	 [0.05, 0.80, 0.55]]  # step    (x^6)
)

x_2 = inputs[1]   # journey (x^2)
d_in = inputs.shape[1]  # Input dimension (number of features)
d_out = 2  # Output dimension (number of features in the output)

torch.manual_seed(123)  # For reproducibility
W_query = torch.nn.Parameter(torch.rand(d_in, d_out))  # Query weight matrix
W_key = torch.nn.Parameter(torch.rand(d_in, d_out))    # Key weight matrix
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))  # Value weight matrix

#1. Derive the query, key, and value vectors for x^2 (journey) using the weight matrices:
query_2 = x_2 @ W_query  # Query vector for x^2
keys = inputs @ W_key      # Key vector for x^2
values = inputs @ W_value  # Value vector for x^2

#2. Compute the attention scores for x^2 (journey) using the query and key vectors:
keys_2 = keys[1]
attn_scores_2 = torch.matmul( query_2,keys.T)  # Attention scores for x^2
print("Attention scores for x^2:\n", attn_scores_2)

#3. Calculate the attention weights by normalizing the attention scores using the softmax function:
d_k = keys.shape[1]  # Dimension of the key vectors
scaled_attn_scores_2 = torch.softmax(attn_scores_2 / torch.sqrt(torch.tensor(d_k, dtype=torch.float32)), dim=-1)  # Scaled attention scores for x^2( )
print("Scaled attention scores for x^2:\n", torch.sum(scaled_attn_scores_2))

#4. Compute the context vector for x^2 (journey) by taking the weighted sum of the value vectors using the attention weights:
context_vector_2 = torch.matmul(scaled_attn_scores_2, values)  # Context vector for x^2
print("Context vector for x^2:\n", context_vector_2)