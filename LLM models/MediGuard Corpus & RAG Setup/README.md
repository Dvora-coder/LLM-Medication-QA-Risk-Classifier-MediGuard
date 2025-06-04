# MediGuard Corpus & RAG Setup
All LLM-based models in this project utilized the custom medical corpus through a RAG (Retrieval-Augmented Generation) approach 
to enhance their predictions with relevant external knowledge.

The code also generates an "overlap" file, which records passages appearing in multiple source documents to help identify and handle duplicated or highly similar content.

This README explains how to build and index the textual corpus for MediGuard and integrate it into a Retrieval-Augmented Generation (RAG) pipeline. 

---

## Overview

Retrieval-Augmented Generation (RAG) combines a dense retrieval component (e.g., DPR + FAISS) with a large language model to improve question classification by supplying relevant passages from a prepared corpus. The workflow is:

1. **Corpus Creation**: Gather medication-related documents (package inserts, clinical guidelines, FAQs, drug–drug interaction warnings) into a single text corpus.
2. **Embedding Index**: Use a Dense Passage Retriever (DPR) encoder to convert each passage into a fixed-length embedding, then build a FAISS index over these vectors.
3. **Retriever Module**: Given a new question, encode it with the same DPR encoder and query FAISS to fetch top-k most similar passages.
4. **RAG Prompting**: Combine retrieved snippets with a prompt template, then feed into a classification model (e.g., GPT-4.1, BioBERT).
5. **Evaluation**: Compare the RAG-augmented output against ground-truth labels to measure improvements.

---

## Step 1: Collect & Prepare Source Documents

### Identify Data Sources

Gather any reliable, medication-related text sources, such as:

* WHO Essential Medicines Lists (text or PDF)
* DrugBank interaction info (HTML or JSON)

### Preprocess & Clean Text

1. **Load raw files**
2. **Tokenize & split into passages**: break long documents into reasonably sized chunks (e.g., 200–300 words) to preserve context.
3. **Remove boilerplate**: eliminate headers/footers, page numbers, repetitive disclaimers.
4. **Normalize whitespace & punctuation**: lowercase, strip excessive spaces, fix encoding issues.

---

## Step 2: Build the Embedding Index

### Choose and Load a DPR Encoder

We recommend using "facebook/dpr-ctx_encoder-single-nq-base" (or a domain-specific checkpoint).
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
import json
import torch
from tqdm import tqdm

tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model.eval().to("cuda")  # use GPU if available

embeddings = []
ids = []

with open("data/processed_corpus.jsonl", "r") as f:
    for line in tqdm(f, desc="Encoding passages"):
        item = json.loads(line)
        text = item["text"]
        inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        with torch.no_grad():
            vector = model(**inputs).pooler_output.squeeze().cpu().numpy()
        embeddings.append(vector)
        ids.append((item["doc_id"], item["passage_id"]))

### Encode the Corpus into Vectors

### Create & Save a FAISS Index

To build the index:

import faiss
import numpy as np
import pickle

# Load embeddings and ids
embeddings = np.load("data/dpr_embeddings/embeddings.npy")
ids = np.load("data/dpr_embeddings/ids.npy", allow_pickle=True)

# Build FAISS index (e.g., IndexFlatIP for cosine similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Normalize vectors if using inner-product for cosine similarity
faiss.normalize_L2(embeddings)

index.add(embeddings)  

# Save index to disk
faiss.write_index(index, "data/faiss_index/index.faiss")

# Save mapping from vector index → (doc_id, passage_id)
with open("data/faiss_index/id2metadata.pkl", "wb") as f:
    pickle.dump(ids.tolist(), f)

## Step 3: Implement the Retriever

### Load FAISS Index & DPR Encoder

import faiss
import numpy as np
import pickle
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

# Load index and metadata
index = faiss.read_index("data/faiss_index/index.faiss")
with open("data/faiss_index/id2metadata.pkl", "rb") as f:
    id2metadata = pickle.load(f)

# Load question encoder
q_tok = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_enc = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_enc.eval().to("cuda")


### Write a "retrieve_topk(question, k)" Function

def retrieve_topk(question: str, k: int = 3):
    """
    Given a question string, return the top-k passages (doc_id, passage_id, text).
    """
    # Encode question
    inputs = q_tok(question, truncation=True, max_length=64, return_tensors="pt").to("cuda")
    with torch.no_grad():
        q_vector = q_enc(**inputs).pooler_output.cpu().numpy()
    faiss.normalize_L2(q_vector)

    # Search in FAISS
    distances, indices = index.search(q_vector, k)
    topk = []
    for idx in indices[0]:
        doc_id, passage_id = id2metadata[idx]
        # Load the actual passage text from processed_corpus.jsonl
        topk.append({"doc_id": doc_id, "passage_id": passage_id, "text": fetch_passage_text(doc_id, passage_id)})

    return topk

def fetch_passage_text(doc_id, passage_id):
    # Implement a lookup from processed_corpus.jsonl or a local cache
    # Example: store passages in a dict mapping (doc_id, passage_id) → text
    return passage_lookup[(doc_id, passage_id)]

---

## Step 4: Integrate with Your Classification Pipeline

### Construct RAG Prompts

Use retrieved passages to create a context block:

def build_rag_prompt(question: str, retrieved_passages: list, examples: list):
    """
    Combine instruction, few-shot examples, retrieved context, and the test question.
    """
    instruction = (
        "You are a clinical assistant. Given a question about medication, "
        "classify it as 'General' (no urgent risk) or 'Critical' (potentially dangerous). "
        "Use the provided context to inform your decision.\n\n"
    )

    # Format few-shot examples
    example_text = "Examples:\n"
    for q, label in examples:
        example_text += f"- Question: \"{q}\" → {label}\n"
    example_text += "\n"

    # Format retrieved context
    context_text = "Relevant Context:\n"
    for i, passage in enumerate(retrieved_passages, 1):
        context_text += f"{i}. {passage['text']}\n"
    context_text += "\n"

    # Final question prompt
    question_text = f"Question: \"{question}\"\nAnswer:"
    
    return instruction + example_text + context_text + question_text

### Combine Retrieved Context + Prompt Template

from retrieve import retrieve_topk, build_rag_prompt
import openai

# Define few-shot examples (list of (question, label))
few_shot_examples = [
    ("Does warfarin increase bleeding risk when taken with aspirin?", "Critical"),
    ("What is the usual dose of ibuprofen for mild pain?", "General"),
    # add more as needed
]

# Loop over test questions
for question, true_label in test_dataset:  # test_dataset = list of (question, label)
    topk_passages = retrieve_topk(question, k=3)
    prompt_text = build_rag_prompt(question, topk_passages, few_shot_examples)

    # Call GPT-4.1
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.0
    )
    predicted_label = response.choices[0].message.content.strip()
    # Save predicted_label for evaluation

### Call LLM or Classifier

* For GPT-4.1, use the above OpenAI call.
* If you prefer a sequence-classification model (e.g., BioBERT), embed the retrieved passages and original question together, then feed through your model’s forward method.

## Step 5: Evaluate & Iterate

1. **Compare Predictions vs. Ground Truth**

   from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

   y_true = [label for (_, label) in test_dataset]
   y_pred = collected_predictions

   print("Accuracy:", accuracy_score(y_true, y_pred))
   print("F1-score:", f1_score(y_true, y_pred, pos_label="Critical"))
   print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=["General", "Critical"]))

2. **Error Analysis**

   * Examine cases where RAG misclassified "Critical" questions as "General" (false negatives).
   * Check if relevant context passages missed key evidence—consider increasing "k" or retraining DPR.

3. **Iterate on Corpus Quality**

   * Add more specialized documents (e.g., rare drug interactions, FDA bulletins).
   * Re-run preprocessing to encoding to indexing to include new passages.

4. **Refine Prompts / Examples**

   * Experiment with additional few-shot examples covering edge cases.
   * Adjust instruction wording to emphasize risk detection.
