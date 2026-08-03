"""Demonstrate GPT-2 byte-pair encoding."""

#Unknow words are replaced by <unk> token which makes them all look same
#Break words into individual english characters

import tiktoken #GPT 2 tokenizer

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("Hello I found the asdadsadsad qwewqwe couple <|endoftext|> at tea beneath their palm-trees", allowed_special={"<|endoftext|>"})) #Hello I found the couple <|endoftext|> at tea beneath their palm-trees
#50256 reprsents <|endoftext|> token
