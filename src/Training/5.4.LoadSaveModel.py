
import torch

import PreviousChapters

GPT_CONFIG_124M = {
	"vocab_size": 50257,
	"context_length": 1024,
	"emb_dim": 768,
	"n_heads": 12,
	"n_layers": 12,
	"drop_rate": 0.1,
	"qkv_bias": False,
}


model = PreviousChapters.GPTModel(GPT_CONFIG_124M)
torch.save(model.state_dict(), "../../data/model_weights.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = torch.load("../../data/model_weights.pth", map_location=device)
print(loaded_model.keys())