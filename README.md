# 🩺 Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

**Machine Learning Project** for predicting the possibility of **Heart Disease, Diabetes, and Breast Cancer** using patient data.

---

## 📋 Project Objective

This project applies **classification techniques** on structured medical datasets to predict disease risk. It demonstrates end-to-end Machine Learning workflow including data preprocessing, model training, evaluation, and deployment as an interactive web app.

### Key Features
- Supports **3 diseases**: Heart Disease, Diabetes, Breast Cancer
- 4 ML Models: Logistic Regression, SVM, Random Forest, **XGBoost**
- Interactive **Streamlit Web Application**
- Proper data preprocessing (handling categorical & missing values)
- Model saving and loading using Joblib

---

## 📊 Datasets Used

<img width="668" height="169" alt="image" src="https://github.com/user-attachments/assets/d9b8855d-60d5-402a-9e8f-2aa935302b49" />


## 🛠️ Technologies & Tools

- **Programming**: Python
- **ML Libraries**: scikit-learn, XGBoost, pandas, numpy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: Streamlit
- **Others**: Joblib, LabelEncoder, SimpleImputer

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/disease-prediction-ml.git
cd disease-prediction-ml

2. Install Dependencies
Bashpip install -r requirements.txt
3. Train the Models
Bashpython src/train.py
4. Run the Web App
Bashstreamlit run app/app.py

📈 Model Performance
<img width="449" height="203" alt="image" src="https://github.com/user-attachments/assets/5d82d6cf-e237-427c-818b-c205d49607f5" />
(Results may vary slightly based on train-test split)

📷 Screenshots

🗂️ Project Structure
<img width="698" height="347" alt="image" src="https://github.com/user-attachments/assets/3ac80565-d324-4eb2-a399-dc5f139a71fc" />

⚠️ Disclaimer
This project is created for educational and learning purposes only. The predictions should not be used for real medical diagnosis. Always consult a qualified healthcare professional for medical advice.


⭐ Show Your Support
If you found this project helpful, please give it a STAR ⭐ on GitHub!
