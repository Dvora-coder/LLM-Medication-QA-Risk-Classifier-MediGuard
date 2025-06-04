# Baseline Model Training – MediGuard

This notebook implements the **baseline models** for the MediGuard project, which classifies medication-related questions into two risk levels: **General** and **Critical**.

### Purpose:
To evaluate traditional machine learning classifiers using standard feature engineering techniques as a baseline for comparison with advanced models such as BioBERT or GPT-4.1.

### Contents:
- Loading and preprocessing labeled question data
- Vectorizing text using **TF-IDF**
- Applying **SVD (Singular Value Decomposition)** for dimensionality reduction
- Creating a custom feature: **Critical Similarity Score** based on cosine similarity
- Handling class imbalance using **SMOTE**
- Training and evaluating classic ML models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)
  - SGD Classifier with regularization
- Computing evaluation metrics:
  - Accuracy
  - Precision, Recall, F1-score

### Notes:
This notebook serves as a **baseline comparison** point before moving on to LLM-based solutions.
