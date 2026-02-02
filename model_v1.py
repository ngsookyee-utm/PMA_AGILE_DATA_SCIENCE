import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load
df = pd.read_csv("stroke_data.csv")
if "id" in df.columns:
    df = df.drop(columns=["id"])

X = df.drop(columns=["stroke"])
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

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

# Baseline model (simple, no tuning)
baseline_lr = LogisticRegression(max_iter=2000, class_weight="balanced")

model_v1_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", baseline_lr)
])

model_v1_pipeline.fit(X_train, y_train)

joblib.dump(model_v1_pipeline, "stroke_prediction_model_v1.pkl")
print("✅ Baseline model saved as stroke_prediction_model_v1.pkl")
