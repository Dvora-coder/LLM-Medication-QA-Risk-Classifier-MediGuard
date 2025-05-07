# MediGuard

**MediGuard** is an NLP-based project designed to classify medication-related questions by clinical **risk level**:
**General** (safe) or **Critical** (potentially dangerous).

We aim to improve patient safety by identifying high-risk drug-related queries that may require professional medical attention.
![image](https://github.com/user-attachments/assets/2163963a-b88f-46bb-a521-d90def651501)

## Project Overview

* **Input**: Free-text medication questions (e.g., about dosage, side effects, drug interactions)
* **Output**: Risk classification – `General` or `Critical`
* **Task Type**: Binary text classification (clinical risk prediction)

## Dataset & Annotation

* Based on the [MedInfo2019-QA-Medications](https://github.com/abachaa/Medication_QA_MedInfo2019) dataset
* Manually annotated \~700 questions with a new `Risk_Level` label
* Dataset imbalance handled via **SMOTE** (oversampling `Critical` class)
* Mean question length: \~50 words

## Methodology

* **Text Preprocessing**: Lowercasing, punctuation removal, stopword filtering
* **Vectorization**: TF-IDF, with **dimensionality reduction** using SVD
* **Feature Engineering**:
   Added a `Critical Similarity` score based on cosine distance to high-risk questions
* **Models Evaluated**:

  * Logistic Regression (and SGD with regularization)
  * Support Vector Machine (SVM)
  * Random Forest
  * Gradient Boosting
  * KNN
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
* **Validation**: 80/20 train-test split with optional K-Fold CV

## Key Insights

* Binary classification (General vs Critical) yielded more **stable** results (F1 ≈ 77%)
* Multiclass classification (General, Personal, Critical) suffered from **overfitting**
* Adding semantic similarity features boosted performance on borderline cases

## LLM-based Exploration (Next Steps)

We are exploring **LLM-based classifiers** (e.g., DistilBERT, GPT) for deeper semantic understanding, as well as **Named Entity Recognition (NER)** to extract drug names and medical terms for richer context modeling.
