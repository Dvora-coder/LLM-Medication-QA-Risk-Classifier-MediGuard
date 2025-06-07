# 🧠 MediGuard: LLM-Based Risk Classification of Medication Questions

![MediGuard](https://img.shields.io/badge/MediGuard-Drug%20Risk%20QA-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow)
![LLM](https://img.shields.io/badge/Model-BioBERT%20%7C%20GPT--4.1%20%7C%20BlueBERT-purple)

> Automatically classify free-text medication-related questions into `General` (safe) or `Critical` (dangerous) using traditional ML & advanced LLM models (BioBERT, GPT-4.1, BlueBERT).  
> Designed to **improve patient safety** and **support clinical chatbot systems**.

📊 [Results](#results) • 🚀 [Quick Start](#quick-start) • 📚 [Documentation](#documentation) • 🎯 [Architecture](#architecture)

---

## 🗂️ Table of Contents

- 🎯 [Overview](#overview)
- 🖼️ [Graphical Abstract](#graphical-abstract)
- 🧩 [Problem Statement](#problem-statement)
- 🚀 [Key Contributions](#key-contributions)
- 🧪 [Methodology](#methodology)
  - 🔍 [Data & Preprocessing](#data--preprocessing)
  - 🧠 [Models Used](#models-used)
- 📊 [Results](#results)
- 🧠 [Insights](#insights)
- 📦 [Installation](#installation)
- 🚀 [Quick Start](#quick-start)
- 📚 [Documentation](#documentation)
- 👥 [Authors](#authors)
- 📌 [Citation](#citation)

---

## 🎯 Overview

**MediGuard** addresses a critical need in clinical AI:  
Patients frequently ask medication-related questions via chatbots, health forums, and public platforms like **Reddit**, where unverified answers often spread unchecked. While some questions are harmless, others may hint at **life-threatening risks** (e.g., overdose, dangerous interactions) — yet go unnoticed.

This project builds a **binary risk classifier** (`General` / `Critical`) to automatically **flag high-risk questions** and **prioritize medical intervention**.

Why it matters:

- ❗ Prevents clinical harm by identifying unsafe self-medication patterns  
- 💬 Reduces reliance on **non-expert advice** shared in forums and social media  
- 🏥 Helps **health systems and insurers** avoid unnecessary ER visits and treatments  
- 💸 Reduces **financial waste** from avoidable hospitalizations and misprioritized care

By embedding LLM-based risk awareness into QA systems, **MediGuard** improves safety and efficiency across the entire healthcare ecosystem.

---

## 🖼️ Graphical Abstract

**Pipeline Diagram:**
![image](https://github.com/user-attachments/assets/2163963a-b88f-46bb-a521-d90def651501)
**Pipeline Diagram:**
![image](https://github.com/user-attachments/assets/baa95e09-f348-41e5-b5a4-f1e1b552d187)

---

## 🧩 Problem Statement

> **Task**: Classify free-text questions about medications by their risk level  
> **Input**: "Is it safe to take ibuprofen with warfarin?"  
> **Output**: `"Critical"`

**Challenges**:
- Ambiguous, short, layman phrasing
- High precision needed to avoid clinical harm
- Imbalanced dataset (few `Critical` examples)
- Brand names & context variation

---

## 🚀 Key Contributions

✅ Built a full classification pipeline using classical ML and LLMs  
✅ Engineered a novel `Critical Similarity` feature using TF-IDF  
✅ Fine-tuned BioBERT & BlueBERT for domain-specific classification  
✅ Applied RAG-based retrieval to enhance all LLM models (BioBERT, BlueBERT, GPT-4.1) with contextual knowledge
✅ Generated synthetic `Critical` questions to improve rare-class recall

---

## 🧪 Methodology

### 🔍 Data & Preprocessing

| Source | MedInfo2019-QA-Medications |
|--------|-----------------------------|
| Size | 652 questions after cleaning |
| Labels | `General` (522), `Critical` (130) |

- TF-IDF vectorization
- Cosine similarity → `Critical Similarity` score
- SMOTE oversampling for balance
- Dimensionality reduction (SVD)

### 🧠 Models Used

#### 🔸 Baseline Models:
- Logistic Regression
- SVM (best baseline)
- Random Forest
- Gradient Boosting
- SGD (with L2 regularization)
- KNN (poor performance)

#### 🔹 LLMs (with RAG context for all):

- **BioBERT** (`dmis-lab/biobert-base-cased-v1.1`)  
  Fine-tuned on domain data + enriched with RAG-retrieved context passages

- **BlueBERT** (`bionlp/bluebert_pubmed_mimic_uncased`)  
  Fine-tuned similarly to BioBERT, using RAG for medical context integration

- **GPT-4.1** via Azure API  
  Inference-based classification using structured prompts + top-3 passages from RAG retrieval

> ℹ️ **All LLMs used RAG** (Retrieval-Augmented Generation) to access a medical corpus and retrieve context passages via DPR + FAISS. This ensured fair comparison and improved understanding of high-risk questions.


#### 🔹 RAG Pipeline:
- FAISS + DPR encoders for semantic retrieval
- Knowledge corpus: [DrugBank DDI](https://go.drugbank.com/drug-interaction-checker) + [WHO EML](https://www.who.int/groups/expert-committee-on-selection-and-use-of-essential-medicines/essential-medicines-lists)

---

## 📊 Results

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| **SVM (baseline)** | 0.84 | 0.80 |
| Logistic Regression| 0.76 | 0.77 |
| Gradient Boosting | 0.79 | 0.77 |
| Random Forest | 0.68 | 0.70	|
| SGD Logistic (L2) | 0.79 | 0.79 |
| KNN | 0.10 | 0.19 |
| BioBERT (best) | 0.92 | 0.90 |
| BioBERT (best), weighted | 0.9180 | 0.9187 |
| BlueBERT | 0.92 | 0.90 |
| BlueBERT, weighted | 0.9180 | 0.9180 |
| GPT-4.1 (classify) | 0.8702 | 0.7796 |
| GPT-4.1 (gen+classify) | 0.8926 | 0.8506 |

✅ BioBERT achieved the best overall performance (Accuracy 0.92, F1 0.90)

📈 SVM was the strongest classical model

📊 GPT-4.1 with generation improved performance on rare cases

❌ KNN was ineffective due to lack of generalization on Critical class

---

## 🧠 Insights

- The custom Critical Similarity feature (cosine similarity over TF-IDF) provided a useful semantic signal to flag borderline-risk questions.
- RAG-based context retrieval consistently improved model performance across all LLMs.
- GPT-4.1 achieved stronger overall performance after synthetic data augmentation, with Macro-F1 rising from 0.78 to 0.85, and Accuracy from 0.87 to 0.89.
- BioBERT delivered the best results, benefiting from its biomedical-domain pretraining — making it highly aligned with medication-related QA phrasing.

---

## 🚀 Quick Start

Run baseline model training:

```bash
jupyter notebook notebooks/MediGuard_baseline_training.ipynb
```

Run LLM fine-tuning:

```bash
jupyter notebook notebooks/BioBERT.ipynb
```

Run GPT-4.1 classification (Azure key required):

```bash
jupyter notebook notebooks/GPT4_1.ipynb
```

---

## 📚 Documentation

See each notebook under `/notebooks/` for:

* Preprocessing
* Feature engineering
* Classical ML
* BioBERT / BlueBERT fine-tuning
* GPT-4.1 inference & generation

---

## 🗂️ Project Structure

```
MediGuard/
├── data/
│ ├── MedInfo2019-QA-Medications.xlsx # Original QA dataset
│ ├── DDI_data.xlsx # Drug–drug interaction data (DrugBank)
│ ├── EML_export.xlsx # WHO essential medicines data
│ └── knowledge_corpus.csv # Combined corpus for RAG
│
├── notebooks/
│ ├── 01_EDA_Preprocessing.ipynb # Data cleaning, label analysis, SMOTE
│ ├── MediGuard_baseline_training.ipynb # Classical ML models (TF-IDF, SVM, etc.)
│ ├── BioBERT.ipynb # BioBERT fine-tuning with RAG context
│ ├── BlueBERT.ipynb # BlueBERT fine-tuning with RAG context
│ └── GPT4_1.ipynb # GPT-4.1 prompt engineering + generation
│
├── models/
│ ├── bio/ # BioBERT checkpoints
│ └── blue/ # BlueBERT checkpoints
│ ├── gpt4_preds        # GPT-4.1 predictions via Azure OpenAI API (no local model)
│
└── README.md # Project documentation
```
---

## 👥 Authors

Dvora Goncharok & Arbel Shifman  
Final-year B.Sc. students in Digital Medical Technologies  
Holon Institute of Technology (HIT), Israel

---

## 📚 References & Technical Stack

### 📝 Key Papers & Related Work

* **[PLOS ONE (2020)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0230876)** – Identifying medication-related intent in patient questions using NLP techniques
* **[JMIR MedInform (2022)](https://medinform.jmir.org/2022/9/e37770)** – Risk-level classification of COVID-related telehealth queries with SBERT
* **[Kütahya Med J (2023)](https://dergipark.org.tr/en/pub/kutfd/issue/81699/1369468)** – ChatGPT performance on triage decision-making
* **[JMIR mHealth (2023)](https://mhealth.jmir.org/2023/1/e49995)** – Evaluation of GPT-based triage in emergency medicine
* **[Turkish Journal of Emergency Medicine (2023)](https://journals.lww.com/tjem/fulltext/2023/23030/performance_of_emergency_triage_prediction_of_an.4.aspx)** – ChatGPT’s strengths and limitations in identifying emergency severity
* **[ScienceDirect (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0735675724000664)** – Diagnostic and triage accuracy of various medical AI tools

---

### 🛠️ Technical Stack

* **Models**: BioBERT, BlueBERT, GPT-4.1 (with and without synthetic generation)
* **Retrieval**: DPRContextEncoder, FAISS indexing (IVFFlat)
* **Augmentation**: GPT-4.1 prompted generation of synthetic `Critical` questions
* **Vectorization**: TF-IDF, SVD (dimensionality reduction), Critical Similarity scoring
* **Libraries**: PyTorch, HuggingFace Transformers, scikit-learn, imbalanced-learn, FAISS
* **Platform**: Google Colab (Python 3.8, Tesla T4 GPU), Azure OpenAI API

---

### 🙏 Acknowledgments

 **MedInfo2019-QA-Medications** dataset — for providing the core question set used in this project  
  *(Original creators: Aansori et al., shared via GitHub [here]https://github.com/abachaa/Medication_QA_MedInfo2019)*
    Aansori et al. (2019). MedInfo2019-QA-Medications. Available on GitHub
* **DrugBank** – for enabling reference to structured drug–drug interaction data
* **WHO Essential Medicines List (EML)** – for grounding domain corpus
* **Microsoft Azure** – for providing GPT-4.1 access through student credits
* **Holon Institute of Technology** – for academic support of this project

---

⭐ If you found this work useful or inspiring, please consider starring the repo or citing us!
For questions, feedback, or collaboration — feel free to [open an issue](https://github.com/Dvora-coder/LLM-Medication-QA-Risk-Classifier-MediGuard/issues).

---

## 📌 Citation
> 📌 If you use this work, please cite it as follows:
```
@misc{MediGuard2024,
  author = {Goncharok, Dvora and Shifman, Arbel},
  title = {MediGuard: Risk Classification of Medication Questions Using LLMs},
  year = {2024},
  url = {https://github.com/Dvora-coder/LLM-Medication-QA-Risk-Classifier-MediGuard}
}
```
