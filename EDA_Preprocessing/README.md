# MediGuard EDA

This notebook ("EDA_MediGuard_.ipynb") performs exploratory data analysis on the MediGuard QA dataset. In broad terms, it:

1. **Loads the data** from an Excel file containing medication-related questions and answers.
2. **Cleans and preprocesses** the text:
   - Strips and lowercases question strings.
   - Removes duplicate entries.
3. **Creates basic features**:
   - Counts words in each question (and answer, if present).
   - Records null-value summaries to identify any missing data.
4. **Visualizes key distributions**:
   - Plots how many questions exist per category (e.g., "General" vs. "Critical").
   - Shows frequency of top words across all questions.
5. **Drops unused columns** so that only the essential fields (question text and risk label) remain.
6. **Builds a final DataFrame** containing just "Question" and "Risk_Level" for downstream modeling.

To use this notebook:
1. Open it in Google Colab or Jupyter.
2. Install required packages:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn nltk sentence-transformers
