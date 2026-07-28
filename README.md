# SLM
Small Language Model from Scratch

For original code refer - https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md

This repo holds code to build a small language model from Scratch
![alt text](FullPipeline.png)

Flow:
=============
TokenizeEmbeddings
    Input text -> Tokenized Text-> Token IDs -> Token Embeddings -> Positional Embeddings -> Input Embeddings
    Chapter 2
    ![alt text](Ch2.png)

Attention Mechanism
    1.Simplified Self-attention-> 2. Self-Attention -> 3. Causal Attention -> 4. Multi-Head Attention
    ![alt text](Ch3.png)

LLM architecture
    ![TransformerBlock](Ch4-TransformerBlock.png)
    ![TransformerPipeline](Ch4-TransformerPipeline.png)
    4_2. Layer Normalization
    ![LayerNorm](Ch4-1_LayerNorm.png)
    4_3. TransforerBlockInternals
    ![TransforerBlockInternals](Ch4-TransforerBlockInternals.png)
    4.4 Shortcut Connection
    ![ShortcutConnection](Ch4-ShortcutConnection.png)
    4.5 Transformer Block
    ![TransformerBlock-Detail](Ch4-TransformerBlock-Detail.png)
    4.6 GPT Model
    4.7 Generate Text
    ![GenerateText](Ch4-GenerateText.png)

Training:
    Overview ![Training5.png](Ch5-Training5.png)

To get started
=================
1) install the packages using command
pip install -r requirements.txt