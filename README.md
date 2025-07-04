# HR-Attrition-ML-Deployment
This Streamlit web app predicts whether an employee is likely to leave the company (attrition) based on HR data. It uses a machine learning model trained on features like age, job role, salary, overtime, job satisfaction, and more. Built with scikit-learn, pandas, SMOTE for handling imbalance, and Streamlit for the UI.


## Features
- Employee attrition classification
- Interactive UI built with Streamlit
- Preprocessing with StandardScaler and manual encoding
- Handles class imbalance using SMOTE

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- Seaborn
- Matplotlib
- Joblib

git add requirements.txt
git commit -m "Add joblib to requirements.txt"
git push


## Deployment
Hosted on Streamlit Community Cloud.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
