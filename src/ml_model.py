
import pandas as pd
import joblib
import os
from src.data_processing import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load cleaned data
df = load_data()

# Outlier treatment
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[col] = df[col].clip(lower, upper)

# Churn encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Separate X and y
x = df.drop("Churn", axis=1)
y = df["Churn"]

# Encoding
x = pd.get_dummies(x, drop_first=True, dtype=int)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Models
model1 = LogisticRegression(max_iter=1000, class_weight="balanced")

model2 = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)

# Predictions
y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)

# Model comparison
df_result = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred1),
        accuracy_score(y_test, y_pred2)
    ],
    "Precision": [
        precision_score(y_test, y_pred1, zero_division=0),
        precision_score(y_test, y_pred2, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, y_pred1, zero_division=0),
        recall_score(y_test, y_pred2, zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred1, zero_division=0),
        f1_score(y_test, y_pred2, zero_division=0)
    ]
})

print(df_result)

# Save best model and preprocessing
os.makedirs("models", exist_ok=True)

best_model = model2

joblib.dump(best_model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(x.columns.tolist(), "models/columns.pkl")

print("Model saved successfully!")
df_result.to_csv("models/model_comparison.csv", index=False)