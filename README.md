# MediGuard

**MediGuard** is an NLP-based classification project that identifies the **clinical risk level** of medication-related questions.
The goal is to distinguish between questions that are routine (**General**) and those that may require medical intervention (**Critical**), thus improving patient safety in medication-related chat systems.

![image](https://github.com/user-attachments/assets/2163963a-b88f-46bb-a521-d90def651501) 

## Project Overview

* **Task**: Binary classification — `General` (safe) vs `Critical` (potentially harmful)
* **Input**: Free-text question (e.g., about dosage, interactions, or side effects)
* **Output**: Risk level classification
* **Use Case**: Enhancing the safety of medication-related question answering in digital health platforms

## Dataset Description

Based on the [MedInfo2019-QA-Medications](https://github.com/abachaa/Medication_QA_MedInfo2019) dataset, we added a custom annotation layer to enable clinical risk assessment.

### Dataset Fields:

| Field Name       | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `Question`       | Free-text medication question (e.g., about dosage, side effects, etc.)      |
| `Question_Type`  | (Original) Category of the question topic (e.g., dosage, administration)    |
| `Drug_Focus`     | (Original) Drug(s) mentioned in the question                                |
| `URL_Source`     | (Original) Source URL for the question                                      |
| `Risk_Level`:    | **Manually added** — one of:                                                |
|                  |   `General`: routine/low-risk question                                      | 
|                  |  `Critical`: potentially dangerous, requires medical attention              |
                                                                  

* **Samples**: \~700 manually annotated questions
* **Imbalance**: The dataset was skewed toward `General`; we applied **SMOTE** to balance it


## Methodology & Pipeline

1. **Text Preprocessing**

   * Lowercasing, punctuation removal, stopword filtering

2. **Vectorization**

   * TF-IDF used to transform text into numerical features
   * **SVD** (TruncatedSVD) applied for dimensionality reduction

3. **Feature Engineering**

   * Created a custom **Critical Similarity** score: cosine similarity to a subset of high-risk questions

4. **Model Training & Evaluation**

   * Models tested:
     `Logistic Regression`, `SVM`, `Random Forest`, `Gradient Boosting`, `KNN`, `SGD`
   * **Evaluation Metrics**:
     Accuracy, Precision, Recall, F1-Score, Confusion Matrix
   * **Validation**: 80/20 train-test split, with optional K-Fold cross-validation


## Key Results & Insights

* **Best Performance**:
  `SVM` and `Logistic Regression` performed best on binary classification (\~77% F1), with no overfitting
* **Multiclass Limitation**:
  Performance dropped significantly when trying to classify `Critical`, `Personal`, and `General` classes (\~53% accuracy)
* **Critical Similarity Feature**:
  Helped detect borderline risky questions, enriching the model’s semantic sensitivity
* **Data Challenges**:
  High feature-to-sample ratio and class imbalance required careful handling



## LLM-Based Modeling

To enhance performance beyond traditional classifiers, we explored **pretrained biomedical language models** and retrieval-augmented methods.

### 1. **BioBERT & BlueBERT Fine-Tuning**

* We fine-tuned domain-specific LLMs on the binary risk classification task:

  * **BioBERT** (pretrained on PubMed/PubMedCentral)
  * **BlueBERT** (pretrained on MIMIC-III and PubMed)
* Training setup:
  "AdamW", learning rate "2e-5", batch size "16", for "3 epochs".
* Evaluation on the test set showed **BioBERT consistently outperformed other models**, especially in detecting "Critical" questions.

### 2. **GPT-4.1 Classification (RAG-based)**

* We used **GPT-4.1** for classification:

  * Questions were combined with up to 3 retrieved context passages from a FAISS index over a custom drug corpus.
  * The model classified each input as "General" or "Critical".
* This method leveraged **Retrieval-Augmented Generation (RAG)** to inject domain-specific context without fine-tuning the LLM.

### 3. **GPT-4.1 Question Generation (Data Augmentation)**

* To address **data scarcity**, we prompted GPT-4.1 to generate \~50 synthetic "Critical" questions (e.g., overdose, dangerous combinations).
* These were manually reviewed and added to the training set.
* Retraining models with the augmented data improved **Critical recall**.



Here’s a compact version you can include right after *Key Results*:

> **LLM Takeaway**:
> Fine-tuned **BioBERT** achieved the best overall accuracy.
> **GPT-4.1 + RAG** offered flexibility without retraining, and **synthetic augmentation** via GPT-4.1 helped address class imbalance and improved sensitivity to dangerous cases.
