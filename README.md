# Efficient and Scalable Estimation of Tool Representations in Vector Space

**Abstract:**
Recent advancements in function calling and tool use have significantly enhanced the capabilities of large language models (LLMs) by enabling them to interact with external information sources and execute complex tasks. However, the limited context window of LLMs presents challenges when a large number of tools are available, necessitating efficient methods to manage prompt length and maintain accuracy. Existing approaches, such as fine-tuning LLMs or leveraging their reasoning capabilities, either require frequent retraining or incur significant latency overhead. A more efficient solution involves training smaller models to retrieve the most relevant tools for a given query, although this requires high quality, domain-specific data. To address those challenges, we present a novel framework for generating synthetic data for tool retrieval applications and an efficient data-driven tool retrieval strategy using small encoder models. Empowered by LLMs, we create ToolBank, a new tool retrieval dataset that reflects real human user usages. For tool retrieval methodologies, we propose novel approaches: (1) Tool2Vec: usage-driven tool embedding generation for tool retrieval, (2) ToolRefiner: a staged retrieval method that iteratively improves the quality of retrieved tools, and (3) MLC: framing tool retrieval as a multi-label classification problem. With these new methods, we achieve improvements of up to 27.28 in Recall@K on the ToolBench dataset and 30.5 in Recall@K on ToolBank. Additionally, we present further experimental results to rigorously validate our methods. For more details, please check out our paper [here](https://arxiv.org/abs/2201.00000).

---
## Installation

1. Create a conda environment and install the dependencies
```
conda create --name ToolRAG python=3.10 -y
conda activate ToolRAG
```

2. Clone and install the dependencies
```
git clone https://github.com/SuhongMoon/ToolRAG.git
cd ToolRAG
pip install -e .
pip install -r requirements.txt
```

---
## Basic Runs

### Generate Synthetic Data
Refer to `toolrag/data_generation/README.md` for more details [here](toolrag/data_generation/README.md).

### Fine Tuning Embedding Models
Refer to `toolrag/query2query/README.md` for more details [here](toolrag/query2query/README.md).

### Generate Tool2Vec Embeddings
Refer to `toolrag/tool2vec/README.md` for more details [here](toolrag/tool2vec/README.md).

### Train MLC Model
Refer to `toolrag/mlc/README.md` for more details [here](toolrag/mlc/README.md).

### Train ToolRefiner Model
Refer to `toolrag/toolrefiner/README.md` for more details [here](toolrag/toolrefiner/README.md).

## Citation
