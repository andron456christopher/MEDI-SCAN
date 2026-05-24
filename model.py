import pandas as pd
import numpy as np

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# LOAD DATA
# -----------------------------

data = pd.read_csv("Data/Training.csv", low_memory=False)

# Target column
y = data["prognosis"].astype(str).str.strip()

# Features
X_raw = data.drop("prognosis", axis=1)

# Convert binary → text (for TF-IDF)
X_text = X_raw.apply(lambda row: " ".join(X_raw.columns[row == 1]), axis=1)

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN ML MODELS
# -----------------------------

models = {}

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
models["lr"] = lr

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
models["rf"] = rf

svm = SVC(probability=True)
svm.fit(X_train, y_train)
models["svm"] = svm

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
models["ada"] = ada

# -----------------------------
# MODEL ACCURACY
# -----------------------------

model_accuracy = {}
for name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    model_accuracy[name] = round(acc * 100, 2)

print("✅ Model Accuracy:", model_accuracy)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_disease(user_input, model_name="rf"):
    try:
        text = str(user_input).lower()
        vector = vectorizer.transform([text])

        model = models.get(model_name, models["rf"])
        disease = model.predict(vector)[0]
        confidence = np.max(model.predict_proba(vector)) * 100

        return disease, round(confidence, 2), model_accuracy

    except Exception as e:
        print("MODEL ERROR:", e)
        return "No prediction", 0, {}
