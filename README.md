# MediGuard

**MediGuard** is an NLP-based classification project that identifies the **clinical risk level** of medication-related questions.
The goal is to distinguish between questions that are routine (**General**) and those that may require medical intervention (**Critical**), thus improving patient safety in medication-related chat systems.
MediGuard leverages both classical ML and state-of-the-art LLMs (BioBERT, BlueBERT, GPT-4.1 + RAG) to automatically flag potentially dangerous medication queries.

![image](https://github.com/user-attachments/assets/2163963a-b88f-46bb-a521-d90def651501) 

![image](https://github.com/user-attachments/assets/baa95e09-f348-41e5-b5a4-f1e1b552d187)


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
                                                                  


# MediGuard: LLM-Based Medication Question Risk Classifier

![Auto](https://img.shields.io/badge/MediGuard-Drug%20Safety-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Transformers](https://img.shields.io/badge/Transformers-4.38.2-yellow)

> NLP-based classification of medication-related questions into `General` and `Critical` risk levels using LLMs (BioBERT, BlueBERT, GPT-4.1)

📊 [Results](#results) • 🚀 [Quick Start](#quick-start) • 📚 [Documentation](#documentation) • 🎯 [Architecture](#architecture)

