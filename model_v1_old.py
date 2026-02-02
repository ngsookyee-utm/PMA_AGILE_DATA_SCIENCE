import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from sklearn.utils.class_weight import compute_class_weight  # Import to compute class weights
import numpy as np  # Import numpy to create array for classes

# Load the dataset
df = pd.read_csv('stroke_data.csv')

# Drop the 'id' column
df = df.drop(columns=['id'])

# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns with the mean strategy
imputer_num = SimpleImputer(strategy='mean')
df[numerical_columns] = imputer_num.fit_transform(df[numerical_columns])

# Impute missing values for categorical columns with the 'most_frequent' strategy
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])

# Encode categorical columns
label_encoder = LabelEncoder()
categorical_columns = [
    'gender',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status'
]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split features and target
X = df.drop(columns=['stroke'])
y = df['stroke']

# Compute class weights to handle class imbalance
# Convert [0, 1] to a numpy array
classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}  # Class 0: No Stroke, Class 1: Stroke

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model with class weights
log_reg = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results (TEXT ONLY)
print(f"Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)

# Save the trained Logistic Regression model
joblib.dump(log_reg, "stroke_prediction_model_v1.pkl")
print("Baseline model saved as stroke_prediction_model_v1.pkl")
