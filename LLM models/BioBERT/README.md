# BioBERT Fine-Tuning Notebook

This notebook contains the training process for **BioBERT**, a language model pretrained on biomedical text (PubMed and PMC), applied to the **MediGuard** project.

## Goal:
Classify free-text medication questions into:
- **General** – safe or routine inquiries
- **Critical** – dangerous or high-risk scenarios

## Main Steps:
- Load the labeled dataset
- Tokenize the questions using BioBERT tokenizer
- Fine-tune the BioBERT model with:
  - Optimizer: "AdamW"
  - Learning rate: "2e-5"
  - Batch size: "16"
  - Epochs: "3–8"
- Evaluate the model using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Output:
- Metrics printed after each training epoch

## Notes:
- BioBERT helps capture domain-specific medical language better than general models
- This notebook complements the baseline and BlueBERT models
