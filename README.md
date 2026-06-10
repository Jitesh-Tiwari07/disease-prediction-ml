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







# 🩺 Disease Prediction System Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

The Disease Prediction System is a Machine Learning-based web application designed to predict the likelihood of diseases using patient medical data. The project leverages classification algorithms to analyze health parameters and provide accurate predictions through an interactive user interface.

This system assists in early disease risk assessment by utilizing structured healthcare datasets and machine learning models.

---

## 🎯 Objectives

* Predict diseases based on patient medical information.
* Compare the performance of multiple classification algorithms.
* Provide a user-friendly interface for real-time predictions.
* Visualize model performance and evaluation metrics.

---

## ✨ Features

* Data preprocessing and cleaning
* Feature scaling and normalization
* Multiple Machine Learning models
* Disease prediction in real time
* Interactive Streamlit dashboard
* Performance evaluation and comparison
* Confusion matrix visualization
* Accuracy and classification reports

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Streamlit
* XGBoost

### Tools

* Jupyter Notebook
* Git
* GitHub

---

## 📂 Project Structure

```text
disease-prediction-ml/
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── disease_model.pkl
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   └── evaluate.py
│
├── app/
│   └── app.py
│
├── screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The project can be trained on healthcare datasets such as:

* Heart Disease Dataset
* Diabetes Dataset
* Breast Cancer Dataset

### Sample Features

* Age
* Gender
* Blood Pressure
* Cholesterol Level
* Glucose Level
* BMI
* Medical Symptoms

---

## 🤖 Machine Learning Algorithms

The following models were evaluated:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest Classifier
* XGBoost Classifier

---

## 📈 Model Performance

| Algorithm           | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~75.3%     |
| SVM                 | ~98.2%     |
| Random Forest       | ~87.5%     |


(Results may vary slightly based on train-test split)

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/disease-prediction-ml.git
cd disease-prediction-ml
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python src/train.py
```

---

## 🌐 Run the Web Application

```bash
streamlit run app/app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

---

## 📸 Screenshots

### Application Dashboard

Add screenshots here.

### Model Performance

Add accuracy graphs and confusion matrix here.

---




⭐ If you found this project useful, consider giving it a star on GitHub.
