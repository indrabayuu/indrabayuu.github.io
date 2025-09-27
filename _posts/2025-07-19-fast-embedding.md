---
title: Fast Embedding Inference
description: Introduction of fast embedding inference
date: 2025-07-19 18:19:30 +0700
categories: [Tutorial]
tags: [embedding,quantization,onnx,python,rust]
media_subpath: /assets/post/fast-embedding
image:
  path: /thumbnail.jpg
  alt:
comments: true
---

## Background

When working on machine learning systems, especially those involving semantic search or Retrieval-Augmented Generation (RAG), the use of embeddings is nearly unavoidable. Embeddings help translate words, sentences, or even entire documents into vectors, capturing meaning in a machine-readable form.

However, for many developers and researchers, GPU resources aren’t always available or affordable. Running large-scale embedding inference on the cloud can quickly become costly, especially when dealing with high-throughput or on-demand applications.

So how can we achieve *blazingly fast* embedding inference... **without a GPU**?

## ONNX Runtime

Enter **ONNX (Open Neural Network Exchange),** a format that allows you to run models across platforms and runtimes. When paired with **ONNX Runtime**, you can execute models on CPU with highly optimized performance, thanks to operator fusion, multi-threading, and other graph-level optimizations.

### ✅ Benefits:

- Run on cheap CPU instances, even your own laptop
- Lightweight & production-ready
- No external dependencies like CUDA
- Compatible with common NLP models

## FastEmbed

FastEmbed is a lightweight, fast, Python library built for embedding generation. [`fastembed`](https://qdrant.github.io/fastembed/) (by **Qdrant**) provides a zero-boilerplate way to do embedding inference using ONNX-accelerated models.

### a. Installation

```bash
pip install -U fastembed
```

### b. Example Usage

This time, I use embedding from [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), which is only 0.22 GB model size according to FastEmbed page.

```python
from fastembed import TextEmbedding

fe_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
text = ["how to running embedding using fastembed?"]
vector = fe_model.embed(text)
print(vector)
```

> The supported embedding model is limited. You can see the **supported model** section on their website 
{: .prompt-tip }

## FastText (without ONNX)

If you need fast, language-agnostic word-level embeddings (e.g., for traditional IR tasks), **FastText** still shines. The original [`fasttext`](https://github.com/facebookresearch/fastText) binary is extremely fast, but consuming a lot of RAM as it load the model weight on memory for fast inference.

### a. Installation

**Download the Embedding Weight first**

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```

**Install Python library**

```bash
pip install fasttext
```

### b. Example Usage

```python
import fasttext

ft_model = fasttext.load_model("cc.en.300.bin")
text = "how to running embedding using fasttext?"
vector = ft_model.get_sentence_vector(text)
print(vector)
```

> This 300 dimension weight will consume 7~8 GB of RAM on idle.
{: .prompt-warning }

## Real-World Scenario

During one of my development cycles, I was restricted to using a virtual machine with **no access to a GPU,** a common constraint in budget-conscious or cloud-optimized environments.

That’s when I discovered that **FastEmbed** was a great fit for the job. The project involved building a **Retrieval-Augmented Generation (RAG)** system for an AI application, where fast and accurate embedding inference was essential.

FastEmbed proved to be both **lightweight** and **surprisingly fast**, making it possible to run concurrent inference tasks efficiently, even in a production setting.

However, there were a few trade-offs. The number of supported models is still limited, and since my documents were primarily in **Indonesian**, I needed a **multilingual embedding model** to handle dense vector comparisons effectively. Fortunately, models like `paraphrase-multilingual-MiniLM` still worked like a charm with ONNX Runtime.

With the right configuration, FastEmbed allowed me to scale CPU-based inference reliably and keep the system lean and cost-effective.

> This application running on **AWS EC2** `t3.small` instance (2 CPU and 2 GB of RAM).
{: .prompt-info }

## Final Thoughts

GPU power isn't a necessity for blazing-fast inference anymore.

With tools like `fastembed`, native `fasttext`, and ONNX Runtime, you can deploy fast, efficient, and scalable embedding pipelines, **all on CPU**. Whether you’re building a lightweight semantic search engine, an on-device assistant, or a low-cost RAG backend, ONNX and CPU-optimized inference can keep things fast and affordable.

So next time you're thinking *“I need a GPU for this,”*—maybe you don’t.

## Repository

Here is an example of a **Rust microservice** that leverages the `fastembed` library to generate and serve embeddings. It uses `gRPC` as the communication protocol, making it efficient for applications that need to request embeddings.

[gRPC Embedding Service](https://github.com/indrabayuu/rust-embedding-service)