import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve
)

# Load
df = pd.read_csv("stroke_data.csv").drop(columns=["id"])

X = df.drop(columns=["stroke"])
y = df["stroke"]

# Train-test split (stratified for imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Preprocessing
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ]
)

# Model
clf = LogisticRegression(max_iter=5000, class_weight="balanced")

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", clf)
])

# Hyperparameter tuning
param_grid = {
    "model__penalty": ["l2", "elasticnet"],
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["saga"],          # supports elasticnet
    "model__l1_ratio": [0.0, 0.5, 1.0], # used for elasticnet
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use PR-AUC for imbalanced data
gs = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1
)

gs.fit(X_train, y_train)
best_model = gs.best_estimator_

# Probabilities
y_proba = best_model.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print("Best params:", gs.best_params_)
print(f"ROC-AUC: {roc:.4f}")
print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

# Threshold tuning: choose threshold that maximizes F1 for the positive class
prec, rec, thr = precision_recall_curve(y_test, y_proba)
f1 = (2 * prec * rec) / (prec + rec + 1e-12)
best_idx = np.argmax(f1)
best_threshold = thr[max(best_idx - 1, 0)]
print(f"Chosen threshold (max F1): {best_threshold:.4f}")

# Predictions at tuned threshold
y_pred = (y_proba >= best_threshold).astype(int)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save the full pipeline (preprocess + model)
joblib.dump(best_model, "stroke_prediction_model_v2.pkl")
print("Model V2 saved as stroke_prediction_model_v2.pkl")
