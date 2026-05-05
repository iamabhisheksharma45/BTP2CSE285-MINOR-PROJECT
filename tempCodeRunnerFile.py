# ==========================================
# HEART DISEASE PREDICTION (FINAL VERSION)
# ==========================================

import pandas as pd
import numpy as np

# ===============================
# 1. LOAD DATASET
# ===============================

columns = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal","target"
]

df = pd.read_csv("heart.csv")
# ===============================
# 2. DATA CLEANING
# ===============================

df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.dropna(inplace=True)

# Convert target to binary
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

print("Dataset Loaded & Cleaned ✅")
print(df.head())

# ===============================
# 3. FEATURE SELECTION (BOOST)
# ===============================

from sklearn.feature_selection import SelectKBest, chi2

X = df.drop("target", axis=1)
y = df["target"]

selector = SelectKBest(score_func=chi2, k=12)
X = selector.fit_transform(X, y)

# ===============================
# 4. TRAIN TEST SPLIT
# ===============================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. SCALING
# ===============================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data Preprocessing Done ✅")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

# Split (stratified for balanced classes)
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipelines (feature selection + scaling + model)
pipe_lr = Pipeline([
    ("sel", SelectKBest(chi2, k=12)),
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

pipe_rf = Pipeline([
    ("sel", SelectKBest(chi2, k=12)),
    ("sc", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    ))
])

pipe_ada = Pipeline([
    ("sel", SelectKBest(chi2, k=12)),
    ("sc", StandardScaler()),
    ("clf", AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.4,
        random_state=42
    ))
])

# 🔥 Soft voting (probabilities)
ensemble = VotingClassifier(
    estimators=[('lr', pipe_lr), ('rf', pipe_rf), ('ada', pipe_ada)],
    voting='soft'
)

# Train
ensemble.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = ensemble.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

# Cross-validation (more reliable)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X, y, cv=cv)
print("\nCross Validation Accuracy:", cv_scores.mean())

# ===============================
# 7. EVALUATION
# ===============================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = ensemble.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

# ===============================
# 8. CROSS VALIDATION (IMPRESS PANEL)
# ===============================

from sklearn.model_selection import cross_val_score

scores = cross_val_score(ensemble, X, y, cv=5)

print("\nCross Validation Accuracy:", scores.mean())

# ===============================
# SAMPLE PREDICTION (FIXED)
# ===============================

print("\n===== SAMPLE PREDICTION =====")

sample = np.array([[52,1,2,130,250,0,1,170,0,2.3,2,0,2]])

# 🔥 Apply same feature selection
sample = selector.transform(sample)

# 🔥 Then scaling
sample = scaler.transform(sample)

result = ensemble.predict(sample)

if result[0] == 1:
    print("Prediction: High Risk of Heart Disease ❗")
else:
    print("Prediction: Low Risk of Heart Disease ✅")