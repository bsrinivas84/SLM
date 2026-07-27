# SLM
Small Language Model from Scratch

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
    ![alt text](Ch4-TransformerBlock.png)
    ![alt text](Ch4-TransformerPipeline.png)
    4_2. Layer Normalization
    ![alt text](Ch4-1_LayerNorm.png)
    4_3. TransforerBlockInternals
    ![alt text](Ch4-TransforerBlockInternals.png)

To get started
=================
1) install the packages using command
pip install -r requirements.txt