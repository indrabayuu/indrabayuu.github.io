---
title: BERT Fine-Tuning Pipeline
description: The easiest way to fine-tune BERT models for classification tasks.
date: 2025-10-01 21:19:30 +0700
categories: [Tutorial]
tags: [fine-tuning,bert,pipeline,python]
media_subpath: /assets/post/bert-finetuning-pipeline
image:
  path: /thumbnail.jpg
  alt:
comments: true
---

Ever tried fine-tuning BERT and got overwhelmed by all the configurations, hyperparameters, and setup complexity? Yeah, me too. That's why I built **BERT Fine-Tuning Pipeline** - a user-friendly solution that lets you fine-tune any BERT model with just a simple YAML configuration file.

[🚀 Get Started on GitHub](https://github.com/indrabayuu/bert-finetuning-pipeline)

## What Is This?

BERT Fine-Tuning Pipeline is a minimalist framework for fine-tuning BERT models on classification tasks. Whether you're doing sentiment analysis, topic classification, or any text categorization task, this pipeline handles both binary and multi-class classification automatically.

### Key Features

- **Simple Configuration**: Edit a YAML file or Python dictionary - that's it
- **Flexible Data Input**: Support for CSV and XLSX formats with multiple dataset merging
- **Advanced Preprocessing**: Remove symbols, mentions, emojis, text normalization, and more
- **Model Agnostic**: Works with any BERT variant including the newest ModernBERT
- **Smart Label Handling**: Automatic label encoding for both binary and multi-class tasks
- **Two Deployment Options**: Jupyter notebooks for quick experiments, Python scripts for production

## Quick Start

Want to jump right in? Here's the fastest path to training your first model:

### Option 1: Cloud Environment (Google Colab/Kaggle)

1. Clone or download the repository
2. Upload the `notebook/pipeline.ipynb` to your preferred platform (Colab/Kaggle)
3. Upload your dataset (CSV or XLSX)
4. Edit the `config` variable on first cell
5. Run all cells

That's it! The cloud environment already has PyTorch and Transformers pre-installed.

### Option 2: Local Environment

1. **Clone the repository**
    ```bash
    git clone https://github.com/indrabayuu/bert-finetuning-pipeline
    
    cd bert-finetuning-pipeline
    ```

2. **Install uv package manager** (it's blazing fast!)

    Follow the official guide: [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

3. **Install dependencies based on your hardware**
    For NVIDIA GPU:
    ```bash
    uv venv && uv sync --extra cuda
    ```

    For CPU or Apple Silicon (M-series):
    ```bash
    uv venv && uv sync --extra cpu
    ```

4. **Edit Configuration**
    - Jupyter Notebook: Edit the `config` varaible on the first cell.
    - Python Script: Edit `config.yaml` on the root folder.

5. **Run the pipeline**

    sing Jupyter Notebook (recommended for experiments):
    ```bash
    jupyter notebook
    # Open and run the provided notebook
    ```

    Using Python script (recommended for production):
    ```bash
    python main.py
    ```

Now let's dive into the configuration!

## Understanding the Configuration

The pipeline uses a single configuration structure that works identically in both notebook and YAML formats. Let's break it down step by step, following your journey from setup to training.

> I recommend using the notebook version first to get familiar with the pipeline, then switch to the Python script when you're ready for production deployment.
{: .prompt-tip }

### Step 1: Authentication & Credentials

```python
"credentials": {
    "huggingface_token": None,
    "wandb_api_key": None
}
```

**What you need to know:**
- `huggingface_token`: Only required if you're using restricted models (like Llama or some private models). Get yours at [HuggingFace tokens](https://huggingface.co/settings/tokens)
- `wandb_api_key`: Required and highly recommended for experiment tracking. Get it from [Weights & Biases](https://wandb.ai/authorize)

### Step 2: Choose Your Model

```python
"base_model": {
    "model_id": "answerdotai/ModernBERT-base"
}
```

**What you need to know:**
- Use any BERT-based model from HuggingFace
- Popular choices: `bert-base-uncased`, `distilbert-base-uncased`, `answerdotai/ModernBERT-base`
- The model ID is exactly as shown on the HuggingFace model card

**Model selection tips:**
- Starting out? Use `distilbert-base-uncased` (faster, smaller)
- Want best accuracy? Use `bert-large-uncased` or `ModernBERT-large`
- Multilingual task? Try `bert-base-multilingual-cased`

### Step 3: Prepare Your Data

```python
"dataset": [
    {
        "filepath": "data/train.csv",
        "text_column": "review",
        "label_column": "sentiment"
    },
    {
        "filepath": "data/additional.csv",
        "text_column": "text",
        "label_column": "category"
    }
]
```

**What you need to know:**
- You can use multiple datasets automatically
- Supported formats: CSV (recommended) and XLSX
- Each dataset can have different column names - the pipeline handles it

> Make sure to upload your files to the runtime before `Run`! Common error: `FileNotFoundError` because the file path doesn't match where you uploaded it.
{: .prompt-danger }

**Data requirements:**
- Text column: Your input text (tweets, reviews, documents, etc.)
- Label column: Your categories (can be text or numbers)
- The pipeline automatically encodes text labels to numbers

### Step 4: Configure Preprocessing

This is where the magic happens! Preprocessing can make or break your model's performance.

```python
"preprocessing": {
    "default": {
        "lowercase": True,
        "remove_symbols": True,
        "remove_numbers": False,
        "remove_newlines": True
    },
    "normalization": {
        "enabled": True,
        "filepath": "assets/normalization/id-version.json"
    },
    "advanced": {
        "remove_mentions": True,
        "remove_retweets": True,
        "remove_emojis": False
    }
}
```

**Default Preprocessing Options:**

- `lowercase`: Converts "Hello World" → "hello world"
  - *When to use*: Almost always, unless case matters (like detecting proper nouns)
  
- `remove_symbols`: Removes !, @, #, $, %, etc.
  - *When to use*: General text classification. Keep them for sentiment analysis if punctuation matters
  
- `remove_numbers`: Removes all digits
  - *When to use*: When numbers are noise. Keep them if "5 stars" or "2024" matters
  
- `remove_newlines`: Removes \n, \r, \t characters
  - *When to use*: Always recommended for cleaner text

**Text Normalization:**

Normalization fixes common typos and slang. For example: "coooool" → "cool", "u" → "you"

The normalization file is a simple JSON dictionary:
```json
{
    "coooool": "cool",
    "u": "you",
    "ur": "your",
    "gooood": "good",
    "looove": "love"
}
```

> Create language-specific normalization files. The repository includes `id-version.json` for Indonesian, but you can create your own for any language.
{: .prompt-tip }

**Advanced Preprocessing:**

- `remove_mentions`: Removes @username
  - *When to use*: Twitter/social media data where mentions are noise
  
- `remove_retweets`: Removes "RT" indicators
  - *When to use*: Twitter data to avoid duplicate content
  
- `remove_emojis`: Strips all emoji characters
  - *When to use*: When emojis are noise. Keep them for sentiment analysis! 😊😢

### Step 5: Configure Training

```python
"training": {
    "target_columns": {
        "text_column": "text",
        "label_column": "labels"
    },
    "split_ratio": [0.8, 0.1, 0.1],
    "output_dir": ".cache/output"
}
```

**What you need to know:**
- `target_columns`: The pipeline renames your columns internally to standardized names
- `split_ratio`: [train, validation, test] - standard is 80/10/10
- `output_dir`: Where your trained model checkpoints are saved

**Splitting strategies:**
- Small dataset (<1000 samples)? Try [0.7, 0.15, 0.15]
- Large dataset (>10000 samples)? [0.9, 0.05, 0.05]
- No test set needed? Use [0.8, 0.2, 0.0]

### Step 6: Tune Hyperparameters

This is where you optimize your model's learning process.

```python
"hyperparameters": {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "gradient_accumulation_steps": 1,
    "optim": "adamw_torch_fused",
    "fp16": True,
    "dataloader_num_workers": 0
}
```

**Essential Hyperparameters Explained:**

- **epochs**: How many times the model sees your entire dataset
  - Start with 3-5 for most tasks
  - More epochs ≠ better (risk of overfitting)
  - Watch validation loss - if it increases, you've trained too long

- **batch_size**: How many samples processed at once
  - Larger = faster training but needs more memory
  - GPU memory limited? Try 8 or 16
  - Have powerful GPU? Go for 32 or 64
  - Rule of thumb: Largest size that fits in your GPU

- **learning_rate**: How big the model's learning steps are
  - BERT sweet spot: 2e-5 to 5e-5
  - Too high (>1e-4)? Model won't converge
  - Too low (<1e-6)? Training takes forever
  - When in doubt, use 5e-5

- **weight_decay**: Prevents overfitting by penalizing large weights
  - Standard: 0.01
  - Overfitting? Increase to 0.1
  - Underfitting? Decrease to 0.001

- **warmup_steps**: Gradually increases learning rate at start
  - Helps training stability
  - Standard: 10% of total training steps
  - Formula: (total_samples / batch_size) × epochs × 0.1

- **gradient_accumulation_steps**: Simulate larger batch sizes
  - GPU out of memory? Increase this instead of reducing batch_size
  - Steps=2 with batch=8 effectively equals batch=16
  - Useful trick for limited hardware

- **optim**: Optimizer algorithm
  - `adamw_torch_fused`: Fastest on modern GPUs (recommended)
  - `adamw_torch`: Standard choice
  - Leave as default unless you know what you're doing

- **fp16**: Mixed precision training
  - Speeds up training by ~2x
  - Uses less GPU memory
  - Only works on NVIDIA GPUs
  - Set to `False` on CPU or Apple Silicon

- **dataloader_num_workers**: Parallel data loading
  - CPU cores for loading data
  - Google Colab/Kaggle: Keep at 0 (can cause issues)
  - Local with good CPU: Try 4 or 8

### Step 7: Configure Logging & Checkpointing

```python
"loggings": {
    "logging_strategy": "steps",
    "logging_steps": 100,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1"
}
```

**Logging Options:**

- `logging_strategy` & `logging_steps`: Log training metrics every N steps
  - "steps" + 100: See updates every 100 training steps (recommended)
  - "epoch": See updates only at end of each epoch

- `eval_strategy`: When to evaluate on validation set
  - "epoch": After each epoch (recommended for small datasets)
  - "steps": Every N steps (good for large datasets)

- `save_strategy`: When to save model checkpoints
  - "epoch": Save after each epoch (recommended)
  - "steps": Save every N steps (for very long training)

- `save_total_limit`: Maximum checkpoints to keep
  - Saves disk space by keeping only best N models
  - Set to 2-3 for most use cases

- `load_best_model_at_end`: Load best performing checkpoint when done
  - Always set to `True` (you want your best model!)

- `metric_for_best_model`: Which metric determines "best"
  - "f1": Best for imbalanced datasets (recommended)
  - "accuracy": Good for balanced datasets
  - "precision": When false positives are costly
  - "recall": When false negatives are costly

### Advanced Settings (Don't Touch Unless You Know What You're Doing!)

```python
"huggingface": {
    "model_path": ".cache/huggingface"
}
```

This sets where HuggingFace caches downloaded models. Leave it as default.

## Python Script vs Notebook: Which Should You Use?

### Use Jupyter Notebooks When:
If you're experimenting with different models, testing preprocessing combinations, or using cloud platforms like Google Colab or Kaggle, notebooks give you that interactive, cell-by-cell execution that makes iteration fast and visual. You can see your data transformations, training progress, and results all in one place. This approach is ideal when you're still figuring out what works best for your specific task or when you're teaching others how the pipeline works.

### Use Python Scripts When:
If you're moving beyond experimentation into production or automation territory. The script version comes with comprehensive structured logging that captures every step of the training process, making it significantly easier to debug issues when things go wrong. If you need to deploy your model as a service, integrate it into a larger ML pipeline, or run automated experiments on a schedule, the script approach gives you the control and reliability you need. It's also version control-friendly and highly customizable if you need to extend the pipeline's functionality.

## Troubleshooting & FAQ

### Common Issues

**Q: "FileNotFoundError: No such file or directory"**

A: This is the most common error, especially on cloud platforms!

- **Cloud (Colab/Kaggle)**: Did you upload the file to the runtime? Check the file browser on the left
- **Local**: Is the file path relative to where you're running the script?
- **Quick fix**: Use absolute paths like `/content/data/train.csv` (Colab) or full local paths

**Q: "CUDA out of memory"**

A: Your GPU doesn't have enough memory. Try these in order:
1. Reduce `batch_size` (try 8 or even 4)
2. Enable `fp16: True` (if not already)
3. Increase `gradient_accumulation_steps` to 2 or 4
4. Use a smaller model (distilbert instead of bert-large)

**Q: "Model not converging / Loss not decreasing"**

A: Your learning rate might be off:
- Try reducing `learning_rate` to 2e-5
- Increase `warmup_steps` to 500-1000
- Check if your data is properly labeled
- Make sure preprocessing isn't too aggressive (removing too much information)

### Configuration Questions

**Q: Should I enable text normalization?**

A: It depends on your data:
- **Yes**: Social media text, user-generated content with typos/slang
- **No**: Formal documents, academic papers, news articles
- **Tip**: Train one model with and without normalization, compare F1 scores

**Q: When should I remove emojis?**

A: Consider your task:
- **Remove**: Topic classification, formal text categorization
- **Keep**: Sentiment analysis, emotion detection (emojis carry meaning!)
- **Remember**: BERT can learn emoji meanings, so keeping them often helps

**Q: How many epochs should I train?**

A: Start with 3-5 and monitor validation loss:
- Validation loss still decreasing? Add more epochs
- Validation loss increasing but training loss decreasing? You're overfitting - stop training
- Both losses plateaued? You're done, model has converged

**Q: My dataset is imbalanced (90% class A, 10% class B). What should I do?**

A: The pipeline handles this automatically, but you can help:
- Use F1 score as `metric_for_best_model` (already default)
- Consider oversampling minority class before feeding to pipeline
- Check precision and recall separately to understand performance

**Q: Can I use this for multi-label classification?**

A: Not out of the box - this pipeline is designed for multi-class (one label per sample) and binary classification. For multi-label, you'd need to customize the Python script.

**Q: How do I know if my model is good?**

A: Check these metrics:
- **F1 Score > 0.7**: Good for most tasks
- **F1 Score > 0.8**: Very good
- **F1 Score > 0.9**: Excellent (but check for overfitting!)
- **Compare to baseline**: Random guessing gets F1 ≈ 0.5 for binary, lower for multi-class

### Platform-Specific Tips

**Google Colab:**
- Always use GPU runtime (Runtime → Change runtime type → GPU)
- Files uploaded are temporary - they disappear when runtime disconnects
- Save your trained model to Google Drive for persistence

**Kaggle:**
- Enable GPU accelerator in settings
- Kaggle gives you more consistent compute compared to Colab
- Use Kaggle datasets feature for easier data management

**Local Environment:**
- Monitor GPU usage with `nvidia-smi` (NVIDIA) or Activity Monitor (Mac)
- Close other GPU-intensive programs during training
- Consider using `screen` or `tmux` for long training sessions

## What's Next?

You've got the pipeline set up and configured - awesome! Here's what you can do:

1. **Experiment**: Try different models, preprocessing combinations, and hyperparameters
2. **Track Experiments**: Use Weights & Biases to compare different runs
3. **Deploy**: Use the Python script version to serve your model via API
4. **Contribute**: Found a bug or have a feature idea? Open an issue or PR on GitHub!

## Wrapping Up

Fine-tuning BERT doesn't have to be complicated. With this pipeline, you can go from raw data to a trained classifier in minutes, not hours. Whether you're a researcher experimenting with models or a developer building production systems, this pipeline scales with your needs.

The philosophy is simple: **configuration over code**. Change what you need in the YAML file, and let the pipeline handle the complexity.

Happy training!😄