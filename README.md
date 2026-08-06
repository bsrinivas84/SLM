# SLM

## 1. Small Language Model from Scratch

This repository contains code for building a small language model from scratch.
It follows the concepts and implementation presented in
[LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch).

![Full language model pipeline](Images/FullPipeline.png)

## 2. Architecture Flow

### 2.1 Tokenization and Embeddings

1. Input text
2. Tokenized text
3. Token IDs
4. Token embeddings
5. Positional embeddings
6. Input embeddings

![Tokenization and embeddings](Images/1-TokenEmbeddingPipeline.png)

### 2.2 Attention Mechanism

1. Simplified self-attention
2. Self-attention with trainable weights
3. Causal attention
4. Multi-head attention

![Attention mechanism](Images/2-Attention.png)

### 2.3 LLM Architecture

![Transformer block](Images/3-TransformerBlock.png)

![Transformer pipeline](Images/3-TransformerPipeline.png)

#### 2.3.1 Layer Normalization

![Layer normalization](Images/3-1_LayerNorm.png)

#### 2.3.2 Transformer Block Internals

![Transformer block internals](Images/3-TransforerBlockInternals.png)

#### 2.3.3 Shortcut Connections

![Shortcut connection](Images/3-ShortcutConnection.png)

#### 2.3.4 Transformer Block

![Transformer block details](Images/3-TransformerBlock-Detail.png)

#### 2.3.5 GPT Model and Text Generation

![Text generation](Images/3-GenerateText.png)

### 2.4 Loss and Training Loop

![Cross-entropy loss](Images/4-CrossEntropyLoss.png)

![Training loop](Images/4-TrainingLoop.png)

### 2.5 Output Word Derivation

![Last-layer word derivation, part 1](Images/LastLayerWordDerivation1.png)

![Last-layer word derivation, part 2](Images/LastLayerWordDerivation2.png)

## 3. Training

### 3.1 Training Overview

![Training overview](Images/4-Training5.png)

### 3.2 Model Sizes

![Different model sizes](Images/5-ModelSizes.png)

| Model | Parameters | Approximate download size |
| --- | ---: | ---: |
| GPT-2 Small | 124M | 500 MB |
| GPT-2 Medium | 355M | 1.4 GB |
| GPT-2 Large | 774M | 3.1 GB |
| GPT-2 XL | 1,558M | 6.2 GB |

Load a pretrained model with one of the following commands:

```powershell
python 4.5.LoadOpenAIWeights.py --model gpt2-small
python 4.5.LoadOpenAIWeights.py --model gpt2-medium
python 4.5.LoadOpenAIWeights.py --model gpt2-large
python 4.5.LoadOpenAIWeights.py --model gpt2-xl
```

## 4. Fine-Tuning

![Fine-tuning pipeline](Images/5-FineTuning.png)

1. Download and preprocess the dataset, then create data loaders.
2. Initialize the model, load pretrained weights, and adapt it for fine-tuning (Modify last layer).
3. Fine-tune and evaluate the model, then use it to classify new data.
![FineTuning-Layers](Images/5-FineTuning-Layers.png)

## 5. Instruction Fine-Tuning

![Instruction Fine-Tuning](Images\6-Instruction-Fine-Tuning-Pipeline.png)

1. Download and convert it into Instruction, Input, Output format
2. Batching Data set
   ![BatchingDataset](Images\6-BatchingDataset.png) 
3. Adding Data loaders + Formatting
4. Load Pretrained GPT 2- 355M model
5. Fine tuning step using sample data
6. Getting responses using the sample data
7. Scoring the responses using another LLM(Ollama ) as a judge and calculate    
    accuracy.


## 6. Getting Started

### 6.1 Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 6.2 Install Dependencies

```powershell
pip install -r requirements.txt
```