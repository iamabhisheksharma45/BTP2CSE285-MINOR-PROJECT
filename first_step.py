import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("heart.csv")

# ❗ Replace '?' with NaN
df.replace('?', pd.NA, inplace=True)

# ❗ Convert all columns to numeric
df = df.apply(pd.to_numeric)

# ❗ Fill missing values
df.fillna(df.mean(), inplace=True)

# Split
X = df.drop("target", axis=1)
y = df["target"]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
ada = AdaBoostClassifier()

# Ensemble
model = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('ada', ada)],
    voting='soft'
)

model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model Trained Successfully")