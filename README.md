#  Credit Card Fraud Detection App

This project is a machine learning-based web application that detects fraudulent credit card transactions. It is built using Python, scikit-learn, XGBoost, and deployed via Streamlit.

---

## Features

- Trained on a dataset of real credit card transactions
- Supports multiple ML models: Random Forest, Logistic Regression, XGBoost
- Provides classification metrics: precision, recall, confusion matrix
- Streamlit-based web interface for predictions
- Easy deployment with GitHub integration

---

##  Live Demo

[Click here to try the app on Streamlit](https://creditcard-fraud-detection-model-9pzo5brcfwmxdmzsifwfva.streamlit.app/)

---

##  Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit frontend UI for real-time fraud detection |
| `main.py` | Prediction logic and model loader |
| `credit_card.py` | Training pipeline: data loading, EDA, model training, saving `.pkl` |
| `fraud_detection_pipeline.pkl` | Saved trained ML model |
| `fraudTest.csv` | Dataset used for training/testing the model |
| `requirements.txt` | All Python dependencies for the project |
| `.gitattributes` | Git settings for file handling |

---

##  How It Works

1. Load dataset from `fraudTest.csv`
2. Explore the data (EDA), handle class imbalance if any
3. Train models (RandomForest, LogisticRegression, XGBoost)
4. Save the best model as `fraud_detection_pipeline.pkl`
5. Use `main.py` to load the model and predict on new inputs
6. `app.py` provides a web interface using Streamlit

---

## Example Usage (CLI)

If you're testing locally:

```bash
python credit_card.py  # Trains the model
streamlit run app.py   # Launches the web app

## Model Performance

Update with actual metrics

Accuracy: 99%

Precision: 98%

Recall: 95%

## **Dataset**

The project uses a CSV file fraudTest.csv that contains anonymized credit card transaction data.

## Make sure the dataset is in the project root directory before running credit_card.py.

## **Credits**

Built by: Sujatha-Lab

Dataset Source: Kaggle - Credit Card Fraud Detection

##**Libraries Used:**

scikit-learn

xgboost

streamlit

pandas

joblib

## **Installation & Setup (Local)**

1. Clone the repository

```bash
git clone https://github.com/sujatha-lab/CreditCard-Fraud-Detection-Model.git
cd CreditCard-Fraud-Detection-Model

```

2.Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

```
3.Install dependencies
```bash
pip install -r requirements.txt

```

4.Run the app
```bash
streamlit run app.py

```
## Notes

Ensure .venv/ is added to .gitignore

Do not commit sensitive data (real credit card info, if any)

Feel free to fork, modify, and enhance the project

##**License**

This project is for educational purposes. Feel free to use and modify it with credit.

## 🖼️ App Screenshot

Here is a preview of the Credit Card Fraud Detection app:
![App Screenshot](Screenshot.png)


