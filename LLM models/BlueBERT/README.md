# LLM Training – Fine-Tuning Models

This folder contains notebooks for training and fine-tuning **large language models (LLMs)** to classify medication-related questions into two risk levels:  
**General** (safe) or **Critical** (potentially dangerous).

## BlueBERT.ipynb

This notebook fine-tunes the **BlueBERT** model, which is pretrained on medical texts.

### What it does:
- Loads the labeled dataset
- Tokenizes the questions
- Fine-tunes BlueBERT for classification
- Evaluates performance
