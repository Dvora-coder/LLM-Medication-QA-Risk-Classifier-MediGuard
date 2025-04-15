# MediGuard

This project uses LLMs to classify medication-related questions by risk level: **General**, **Personal**, or **Critical**.  
Based on the [MedInfo2019-QA-Medications](https://github.com/abachaa/Medication_QA_MedInfo2019) dataset, we added a custom annotation layer with risk labels.  
We evaluate both traditional ML models and LLM-based approaches (e.g., DistilBERT, GPT) to improve safety in drug-related question answering.

![image](https://github.com/user-attachments/assets/203e90c3-5efc-43c4-83a5-94b8a5e53c16)



## Key Features
- Manual risk-level annotation of ~700 patient questions
- Classification using both traditional and LLM-based models
- Evaluation with accuracy, precision, recall, F1
- Optional drug name extraction (NER) for improved understanding
