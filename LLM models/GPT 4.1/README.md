# GPT-4.1 Inference Notebook

This notebook runs inference using **GPT-4.1** to classify medication-related questions into **“General”** or **“Critical”** categories.

## Goal:
Use GPT-4.1 with a **few-shot, retrieval-augmented prompt** (via prompt engineering) to perform risk classification without any fine-tuning.

## Main Steps:
1. Load a sample of labeled test questions
2. Retrieve relevant context passages for each question 
3. Format each question into a prompt including few-shot examples and retrieved context
4. Send requests to GPT-4.1 via the Azure OpenAI API
5. Collect model predictions ("General" or "Critical")
6. Compare predictions with true labels
7. Evaluate results 

## Notes:
* This notebook does **not** perform any fine-tuning. It uses GPT-4.1 as a ready-to-use classifier.
* Performance depends heavily on the **quality of the few-shot prompt** (including retrieved context).
* Requires an active **Azure OpenAI** subscription and a valid API key.
