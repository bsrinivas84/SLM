# SLM
Small Language Model from Scratch

For original code refer - https://github.com/rasbt/LLMs-from-scratch/blob/main/README.md

This repo holds code to build a small language model from Scratch
![alt text](Images\FullPipeline.png)

Flow:
=============
TokenizeEmbeddings
    Input text -> Tokenized Text-> Token IDs -> Token Embeddings -> Positional Embeddings -> Input Embeddings
    Chapter 2
    ![alt text](Images\Ch2.png)

Attention Mechanism
    1.Simplified Self-attention-> 2. Self-Attention -> 3. Causal Attention -> 4. Multi-Head Attention
    ![alt text](Images\Ch3.png)

LLM architecture
    ![TransformerBlock](Images\Ch4-TransformerBlock.png)
    ![TransformerPipeline](Images\Ch4-TransformerPipeline.png)
    4_2. Layer Normalization
    ![LayerNorm](Images\Ch4-1_LayerNorm.png)
    4_3. TransforerBlockInternals
    ![TransforerBlockInternals](Images\Ch4-TransforerBlockInternals.png)
    4.4 Shortcut Connection
    ![ShortcutConnection](Images\Ch4-ShortcutConnection.png)
    4.5 Transformer Block
    ![TransformerBlock-Detail](Images\Ch4-TransformerBlock-Detail.png)
    4.6 GPT Model
    4.7 Generate Text
    ![GenerateText](Images\Ch4-GenerateText.png)

Cross Entropy Loss:
    ![CrossEntropyLoss](Images\Ch5-CrossEntropyLoss.png)
    ![Training](Images\Ch5-TrainingLoop.png)

Other:
    ![LastLayerWordDerivation1](LastLayerWordDerivation1.png)
    ![LastLayerWordDerivation2](LastLayerWordDerivation2.png)

Training:
    Overview ![Training5.png](Images\Ch5-Training5.png)

To get started
=================
1) install the packages using command
pip install -r requirements.txt