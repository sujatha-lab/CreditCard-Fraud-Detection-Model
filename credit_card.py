



# Data Analysis Tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML tools
from sklearn.preprocessing import LabelEncoder, StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score,
                precision_score,recall_score,
                f1_score,roc_auc_score,
                classification_report, confusion_matrix,
                roc_curve,precision_recall_curve)

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# For Streamlit deployment
import joblib

# warnings
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('fraudTest.csv')

# Top 5 Data
df.head()

# Basic Information on DataFrame
print(df.info())

# Description on Numerical Columns
print(df.describe())

# Class Distribution of Transactions
print(df['is_fraud'].value_counts())

# Visualizations
# Class Distribution of Transactions
plt.figure(figsize=(6,4))
sns.countplot(x='is_fraud', data=df)
plt.title('Class Distribution of Transactions')
plt.show()

# Transaction Amount Distribution
plt.figure(figsize=(8,4))
sns.histplot(df['amt'], bins=50, log_scale=True)
plt.title('Transaction Amount Distribution')
plt.show()

# Fraud by Category
plt.figure(figsize=(10,4))
sns.countplot(x='category', hue='is_fraud', data=df)
plt.xticks(rotation=45)
plt.title('Fraud by Category')
plt.show()

# Fraud by State
plt.figure(figsize=(10,4))
sns.countplot(x='state', hue='is_fraud', data=df)
plt.xticks(rotation=45)
plt.title('Fraud by State')
plt.show()

# Label Encoding categorical features
cat_cols = ['merchant', 'category', 'gender', 'state', 'city']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Drop unnecessary columns for modeling
drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'first', 'last', 'street',
             'city', 'state', 'zip', 'job', 'dob', 'trans_num']
df_model = df.drop(columns=drop_cols)

# Feature Target split
X = df_model.drop('is_fraud', axis=1)
y = df_model['is_fraud']

# Splitting categorical and Numerical for future pipeline
categorical_cols = ['merchant', 'category', 'gender']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Model Hyperparameter grid
param_grids = {
    'LogisticRegression': {'clf__C': [0.01, 0.1, 1, 10]},
    'RandomForest': {'clf__n_estimators': [100, 200], 'clf__max_depth': [5, 10, None]},
    'XGBoost': {'clf__n_estimators': [100, 200], 'clf__max_depth': [5, 10], 'clf__learning_rate': [0.01, 0.1]}
}

# Finding best model and their parameters
results = {}
for name, model in models.items():
    pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=3),
        n_jobs=-1,
        verbose=1
    )

    print(f"Training and tuning {name}...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Best Parameters for {name}: {grid.best_params_}")
    print(f"ROC-AUC on Test set: {auc:.4f}")

    results[name] = {
        'best_estimator': best_model,
        'best_params': grid.best_params_,
        'cv_best_score': grid.best_score_,
        'test_roc_auc': auc,
    }

# 7. Evaluation Reports for best models
for name, res in results.items():
    print(f"\n{name} Performance on Test Set:")
    best_model = res['best_estimator']
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rocauc = roc_auc_score(y_test, y_pred_proba)
confmat = confusion_matrix(y_test, y_pred)

print("Model Performance on Test Set:")
print(f"Accuracy: {acc:.2%}")
print(f"Precision: {prec:.2%}")
print(f"Recall: {rec:.2%}")
print(f"F1-score: {f1:.2%}")
print(f"ROC AUC: {rocauc:.2%}")
print("Confusion Matrix:")
print(confmat)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

# confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(confmat, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8. Save the best model (highest ROC-AUC) for deployment
best_model_name = max(results, key=lambda x: results[x]['test_roc_auc'])
print(f"\nSaving best model: {best_model_name}")
joblib.dump(results[best_model_name]['best_estimator'], 'fraud_detection_pipeline.pkl')


# Creating Pipeline for Raw Data
# Loading Raw data again
df = pd.read_csv('fraudTest.csv')

print(categorical_cols,'\n',numeric_cols)

# Date handling
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
df['age'] = pd.to_datetime('2020-01-01') - pd.to_datetime(df['dob'])
df['age'] = df['age'].dt.days // 365

X_new = df[list(categorical_cols+numeric_cols)]
y_new = df['is_fraud']

# preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ],
    remainder='drop'  # drop columns not listed
)

# trained optimized model(using actual best params)
best_rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Full pipeline
full_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', best_rf_model)
])

# Fit using pipeline
full_pipeline.fit(X_new, y_new)

# Saving pipeline for deployment
joblib.dump(full_pipeline, 'fraud_detection_pipeline.pkl')

# Load the saved pipeline (update filename if different)
model = joblib.load('fraud_detection_pipeline.pkl')

# Prepare a test input record with all required columns
test_data = {
    'merchant': ['fraud_Murray-Smitham'],
    'category': ['grocery_pos'],
    'gender': ['M'],
    'cc_num': [6550399784335736],
    'amt': [29.06],
    'lat': [26.9379],
    'long': [-82.2388],
    'city_pop': [79008],
    'unix_time': [1372488333],
    'merch_lat': [26.384426],
    'merch_long': [-81.275216],
    'hour': [6],
    'dayofweek': [0],
    'age': [36]
}

test_df = pd.DataFrame(test_data)

# Predict label and probability
pred_label = model.predict(test_df)[0]
pred_proba = model.predict_proba(test_df)[0][1]

print(f"Predicted label: {pred_label} (1=Fraud, 0=Genuine)")
print(f"Fraud Probability: {pred_proba:.4f}")

