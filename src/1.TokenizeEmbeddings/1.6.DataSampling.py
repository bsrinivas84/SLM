"""Build sliding-window token datasets and data loaders."""

import tiktoken #GPT 2 tokenizer
import torch
from torch.utils.data import Dataset,DataLoader

with open('../../data/raw/Verdict.txt', 'r',encoding = "utf-8") as file:
    content = file.read()
    print(len(content)) #20479

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(content)
print(len(enc_text)) #5145

enc_sample = enc_text[50:]
print(len(enc_sample)) #5095

context_size = 4
x= enc_sample[0:context_size]
y= enc_sample[1:context_size+1]

print(f"x:{x}")
print(f"y:     {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

#===================================================================================

class GPTDatasetV1(Dataset):
    """Store overlapping token-ID inputs and next-token targets."""
    def __init__(self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int) -> None:
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt,allowed_special={"<|endoftext|>"})
    
        for i in range(0, len(token_ids) - max_length + 1, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt: str, batch_size: int = 2, max_length: int = 256, stride: int = 128, shuffle: bool = True, drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """Return batches of ``(input_ids, target_ids)`` tensors shaped ``(batch, max_length)``."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

with open('../../data/raw/Verdict.txt', 'r',encoding = "utf-8") as file:
    raw_text = file.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
inputs,targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)

# Inputs:
#  tensor([[   40,   367,  2885,  1464],
#         [ 1807,  3619,   402,   271],
#         [10899,  2138,   257,  7026],
#         [15632,   438,  2016,   257],
#         [  922,  5891,  1576,   438],
#         [  568,   340,   373,   645],
#         [ 1049,  5975,   284,   502],
#         [  284,  3285,   326,    11]])
# Targets:
#  tensor([[  367,  2885,  1464,  1807],
#         [ 3619,   402,   271, 10899],
#         [ 2138,   257,  7026, 15632],
#         [  438,  2016,   257,   922],
#         [ 5891,  1576,   438,   568],
#         [  340,   373,   645,  1049],
#         [ 5975,   284,   502,   284],
#         [ 3285,   326,    11,   287]])
