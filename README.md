# MediGuard

**MediGuard** is an NLP-based classification project that identifies the **clinical risk level** of medication-related questions.
The goal is to distinguish between questions that are routine (**General**) and those that may require medical intervention (**Critical**), thus improving patient safety in medication-related chat systems.
MediGuard leverages both classical ML and state-of-the-art LLMs (BioBERT, BlueBERT, GPT-4.1 + RAG) to automatically flag potentially dangerous medication queries.

![image](https://github.com/user-attachments/assets/2163963a-b88f-46bb-a521-d90def651501) 

![image](https://github.com/user-attachments/assets/baa95e09-f348-41e5-b5a4-f1e1b552d187)
1. **Input & Labeling**

   * A free‐text medication question (e.g., "What happens if I accidentally take 50 mg of lisinopril twice?") enters the system.
   * Each question is manually labeled and verified with a `risk_level` field set to either **General** (routine/no immediate danger) or **Critical** (potentially dangerous, requires medical attention).

2. **Exploratory Data Analysis (EDA)**

   * Once all questions have received a `risk_level` label, an EDA step examines overall distributions, class imbalance (e.g., far more "General" than "Critical"), question lengths, word frequencies, and missing values.
   * Insights from EDA guide decisions about cleaning strategies, feature engineering, and how to handle imbalanced classes (e.g., applying SMOTE).
   * The final dataset contains approximately 700 questions, of which roughly 20% labeled as Critical. We applied SMOTE to address this imbalance.

3. **Cleaning & Tokenization**

   * Raw question text (and any associated answer text) is converted to lowercase, stripped of punctuation, and tokenized.
   * Stopwords are removed, and the data is transformed into a format ready for vectorization (e.g., splitting each string into individual tokens).

4. **Baseline Pipeline**

   * **TF–IDF Vectorization**: Each cleaned question is represented as a sparse TF–IDF vector.
   * **Critical‐Similarity Feature**: A custom cosine‐similarity score is computed between each question and a small set of known high‐risk examples, yielding an additional numeric feature.
   * **SMOTE**: Synthetic Minority Over‐Sampling Technique is applied to balance the "Critical" and "General" classes in the training set.
   * These features feed into classical machine‐learning classifiers (Logistic Regression, SVM, Random Forest, Gradient Boosting, KNN, SGD) with an 80/20 stratified train‐test split (and optional k‐fold cross‐validation) to produce baseline performance metrics.

5. **RAG (Retrieval‐Augmented Generation) Pipeline**

   * **Build FAISS Index**: A DPR (Dense Passage Retriever) context encoder is used to embed every passage in a custom drug‐related corpus, then a FAISS index is built over these embeddings.
   * **Encode Question**: At inference time, each incoming question is passed through the DPR question encoder to produce a query embedding.
   * **Retrieve Top‐k Passages**: Using FAISS, the top‐k most semantically similar passages from the corpus are retrieved and returned as "Relevant Context" for that question.

6. **LLM Tuning & Inference**

   * Retrieved context passages are combined with a few‐shot prompt template and sent to a large language model (e.g., GPT-4.1) for classification without further fine‐tuning.
   * Alternatively, domain‐specific transformers (BioBERT, BlueBERT) can be fine‐tuned on the annotated training set.
   * In both cases, the system outputs a final prediction- "General" or "Critical"—for each question.

7. **Training & Evaluation**

   * For the baseline classifiers and the fine‐tuned BERT models, standard metrics (accuracy, precision, recall, F1, confusion matrix) are calculated on the held-out 20% test set (and, if used, cross‐validation folds).
   * For the RAG + GPT-4.1 approach, predictions are compared to ground-truth labels, and classification metrics are reported to assess whether retrieval‐augmented prompting improved performance without model retraining.

---

In summary, the diagram shows two parallel strategies built on the same cleaned and tokenized data:

* a **Baseline** classical ML pipeline (TF–IDF + SMOTE + traditional classifiers), and
* a **Retrieval-Augmented Generation (RAG)** pipeline (DPR + FAISS + GPT-4.1 or fine-tuned BioBERT/BlueBERT).

Both approaches ultimately feed into evaluation of "General" vs. "Critical" classification metrics.

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
