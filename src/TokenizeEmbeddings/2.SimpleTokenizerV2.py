import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        preprocessed = [
            item if item in self.str_to_int 
            else '<|UNK|>' for item in preprocessed

        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


with open('../../data/raw/Verdict.txt', 'r',encoding = "utf-8") as file:
    content = file.read()
    print(len(content)) #20479


result = re.split(r'([,:;?_!"()\']|--|\s)', content)
preprocessed = [item.strip() for item in result if item.strip()]
print(len(preprocessed)) #4500 instead of 4690

all_tokens = sorted(set(preprocessed))
all_tokens.extend(['<|endoftext|>', '<|UNK|>'])  # Add special tokens
vocab_size = len(all_tokens)
print(vocab_size) # 1212 instead of 1130 + 2 special tokens

vocab = {word: idx for idx, word in enumerate(all_tokens)}
tokenizer = SimpleTokenizerV1(vocab)
print(tokenizer.encode("Hello I found the couple at tea beneath their palm-trees"))
print(tokenizer.decode(tokenizer.encode("Hello I found the couple at tea beneath their palm-trees")))