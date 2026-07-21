import os
import re

with open('Verdict.txt', 'r',encoding = "utf-8") as file:
    content = file.read()
    print(len(content)) #20479


result = re.split(r'([,:;?_!"()\']|--|\s)', content)
preprocessed = [item.strip() for item in result if item.strip()]
print(len(preprocessed)) #4500 instead of 4690

preprocessed = sorted(set(preprocessed))
vocab_size = len(preprocessed)
print(vocab_size) # 1212 instead of 1130

vocab = {word: idx for idx, word in enumerate(preprocessed)}
print(vocab)