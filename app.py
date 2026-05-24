"""
MediScan — Advanced AI Healthcare Chatbot
NLP + Machine Learning + Deep Learning
"""

import os
import re
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder

# Load environment variables
load_dotenv()


BASE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
train_df = pd.read_csv(os.path.join(BASE, "Data", "Training.csv"))
y_raw = train_df["prognosis"].str.strip()
X_binary = train_df.drop("prognosis", axis=1)
ALL_SYMPTOMS = list(X_binary.columns)

desc_df = pd.read_csv(
    os.path.join(BASE, "MasterData", "MasterData", "symptom_Description.csv"),
    header=None, names=["disease", "description"]
)
prec_df = pd.read_csv(
    os.path.join(BASE, "MasterData", "MasterData", "symptom_precaution.csv"),
    header=None
)
sev_df = pd.read_csv(
    os.path.join(BASE, "MasterData", "MasterData", "Symptom_severity.csv"),
    header=None, names=["symptom", "weight"]
)

desc_dict = {
    str(r.disease).strip(): str(r.description).strip()
    for _, r in desc_df.iterrows()
}

prec_dict = {}
for _, row in prec_df.iterrows():
    disease = str(row[0]).strip()
    precs = [str(p).strip()
             for p in row[1:] if str(p).strip() not in ("", "nan")]
    prec_dict[disease] = precs

sev_dict = {
    str(r.symptom).strip().lower().replace(" ", "_"): int(r.weight)
    for _, r in sev_df.iterrows()
}

print(f"✅ Loaded {len(train_df)} training rows | {len(ALL_SYMPTOMS)} symptoms | {y_raw.nunique()} diseases")
print(
    f"   Descriptions: {len(desc_dict)} | Precautions: {len(prec_dict)} | Severity scores: {len(sev_dict)}")

# ══════════════════════════════════════════════════════════════
# 2. NLP PIPELINE
# ══════════════════════════════════════════════════════════════


def preprocess(raw_input: str) -> list:
    text = raw_input.lower()
    text = re.sub(r"[^a-z0-9_,\s]", "", text)
    tokens = [t.strip().replace(" ", "_") for t in text.split(",")]
    return [t for t in tokens if t and t != "nan"]


def row_to_text(row) -> str:
    return " ".join(col for col in ALL_SYMPTOMS if row[col] == 1)


corpus = X_binary.apply(row_to_text, axis=1)
tfidf_vec = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf = tfidf_vec.fit_transform(corpus)

print(f"✅ TF-IDF matrix: {X_tfidf.shape}")

# ══════════════════════════════════════════════════════════════
# 3. ENCODE LABELS
# FIX: Encode string disease labels to integers for ANN
# ══════════════════════════════════════════════════════════════
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)   # "Malaria" → 0, "Diabetes" → 1 etc.

# ══════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT — ONE SHARED SPLIT FOR ALL MODELS
# ══════════════════════════════════════════════════════════════
indices = np.arange(len(y_raw))

idx_train, idx_test = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y_raw
)

# Binary matrix — for Random Forest
X_tr_bin = X_binary.iloc[idx_train].reset_index(drop=True)
X_te_bin = X_binary.iloc[idx_test].reset_index(drop=True)

# TF-IDF matrix — for LR, AdaBoost, SVM, ANN
X_tr_tfidf = X_tfidf[idx_train]
X_te_tfidf = X_tfidf[idx_test]

# String labels — for LR, RF, AdaBoost, SVM
y_train_str = y_raw.iloc[idx_train].reset_index(drop=True)
y_test_str = y_raw.iloc[idx_test].reset_index(drop=True)

# Integer labels — for ANN only
y_train_int = y_encoded[idx_train]
y_test_int = y_encoded[idx_test]

print(f"✅ Split — Train: {len(y_train_str)} | Test: {len(y_test_str)}")

# ══════════════════════════════════════════════════════════════
# 5. TRAIN ALL 5 MODELS
# ══════════════════════════════════════════════════════════════
print("\n⏳ Training models (may take 2-3 minutes) …")

# (a) Logistic Regression — TF-IDF, string labels
lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_model.fit(X_tr_tfidf, y_train_str)
print("   [1/5] ✅ Logistic Regression")

# (b) Random Forest — Binary matrix, string labels
rf_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf_model.fit(X_tr_bin, y_train_str)
print("   [2/5] ✅ Random Forest (200 trees)")

# (c) AdaBoost — TF-IDF, string labels (algorithm param removed for newer sklearn)
ada_model = AdaBoostClassifier(
    n_estimators=150, learning_rate=0.8, random_state=42)
ada_model.fit(X_tr_tfidf, y_train_str)
print("   [3/5] ✅ AdaBoost (150 estimators)")

# (d) SVM — TF-IDF, string labels
svm_model = SVC(kernel="linear", C=1.0, probability=True,
                decision_function_shape="ovr", random_state=42)
svm_model.fit(X_tr_tfidf, y_train_str)
print("   [4/5] ✅ SVM (linear kernel)")

# (e) ANN — TF-IDF, INTEGER labels + early_stopping disabled to avoid type error
ann_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=64,
    learning_rate="adaptive",
    max_iter=500,
    early_stopping=False,   # FIX: disabled to avoid isnan TypeError in newer sklearn
    random_state=42,
    verbose=False,
)
ann_model.fit(X_tr_tfidf, y_train_int)   # FIX: integer labels for ANN
print("   [5/5] ✅ ANN — Neural Network (256→128→64)")
print("\n✅ All 5 models trained successfully!\n")

# ══════════════════════════════════════════════════════════════
# 6. EVALUATE ALL MODELS
# ══════════════════════════════════════════════════════════════


def evaluate_str(model, X_test, y_test, name):
    """For LR, RF, AdaBoost, SVM — string labels"""
    preds = model.predict(X_test)
    return {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, preds) * 100, 2),
        "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, preds, average="weighted", zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_test, preds, average="weighted", zero_division=0) * 100, 2),
    }


def evaluate_ann(model, X_test, y_test_int, y_test_str, name):
    """For ANN — predicts integers, convert back to strings for metrics"""
    preds_int = model.predict(X_test)
    preds_str = le.inverse_transform(preds_int)
    return {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test_str, preds_str) * 100, 2),
        "precision": round(precision_score(y_test_str, preds_str, average="weighted", zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test_str, preds_str, average="weighted", zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_test_str, preds_str, average="weighted", zero_division=0) * 100, 2),
    }


MODEL_RESULTS = [
    evaluate_str(lr_model,  X_te_tfidf, y_test_str, "Logistic Regression"),
    evaluate_str(rf_model,  X_te_bin,   y_test_str, "Random Forest"),
    evaluate_str(ada_model, X_te_tfidf, y_test_str, "AdaBoost"),
    evaluate_str(svm_model, X_te_tfidf, y_test_str, "SVM"),
    evaluate_ann(ann_model, X_te_tfidf, y_test_int,
                 y_test_str, "ANN (Neural Network)"),
]

print("─" * 68)
print(f"{'Model':<26} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9}")
print("─" * 68)
for r in MODEL_RESULTS:
    print(f"{r['model']:<26} {r['accuracy']:>8}% {r['precision']:>8}% "
          f"{r['recall']:>8}% {r['f1']:>8}%")
print("─" * 68)

# ══════════════════════════════════════════════════════════════
# 7. PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════
MODEL_MAP = {
    "lr":  (lr_model,  "tfidf",  "str"),
    "rf":  (rf_model,  "binary", "str"),
    "ada": (ada_model, "tfidf",  "str"),
    "svm": (svm_model, "tfidf",  "str"),
    "ann": (ann_model, "tfidf",  "int"),
}


def predict_disease(raw_input: str, model_key: str = "rf") -> dict:
    try:
        tokens = preprocess(raw_input)
        symptom_text = " ".join(tokens)

        model, feat_type, label_type = MODEL_MAP.get(
            model_key, MODEL_MAP["rf"])

        if feat_type == "binary":
            vec = np.zeros(len(ALL_SYMPTOMS))
            for i, sym in enumerate(ALL_SYMPTOMS):
                if sym in tokens:
                    vec[i] = 1.0
            features = pd.DataFrame([vec], columns=ALL_SYMPTOMS)
        else:
            features = tfidf_vec.transform([symptom_text])

        raw_pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        # Decode ANN integer prediction back to disease name
        if label_type == "int":
            disease = le.inverse_transform([int(raw_pred)])[0]
            classes_str = le.inverse_transform(
                np.arange(len(model.classes_))
            )
        else:
            disease = raw_pred
            classes_str = model.classes_

        confidence = round(float(np.max(proba)) * 100, 2)

        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {"disease":    str(classes_str[i]),
             "confidence": round(float(proba[i]) * 100, 2)}
            for i in top3_idx
        ]

        # Lookup precautions
        precautions = prec_dict.get(disease, [])
        if not precautions:
            for k in prec_dict:
                if k.lower().strip() == disease.lower().strip():
                    precautions = prec_dict[k]
                    break
        if not precautions:
            precautions = ["Consult a qualified doctor immediately"]

        # Lookup description
        description = desc_dict.get(disease, "")
        if not description:
            for k in desc_dict:
                if k.lower().strip() == disease.lower().strip():
                    description = desc_dict[k]
                    break
        if not description:
            description = f"{disease} — please consult a healthcare professional."

        sev_scores = {sym: sev_dict.get(sym, 1) for sym in tokens}
        avg_severity = round(sum(sev_scores.values()) /
                             max(len(sev_scores), 1), 2)

        return {
            "disease":      disease,
            "confidence":   confidence,
            "top3":         top3,
            "precautions":  precautions,
            "description":  description,
            "severity":     sev_scores,
            "avg_severity": avg_severity,
            "consult":      avg_severity >= 5,
            "model_used":   model_key.upper(),
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {
            "disease": "Prediction error", "confidence": 0,
            "top3": [], "precautions": ["Please consult a doctor"],
            "description": "", "severity": {}, "avg_severity": 0,
            "consult": False, "model_used": model_key.upper()
        }


# ══════════════════════════════════════════════════════════════
# 8. FLASK APP
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", symptoms=ALL_SYMPTOMS)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    raw = body.get("symptoms", "")
    model_key = body.get("model", "rf")
    return jsonify(predict_disease(raw, model_key))


@app.route("/model-stats")
def model_stats():
    return jsonify(MODEL_RESULTS)


@app.route("/symptoms")
def symptoms():
    return jsonify(ALL_SYMPTOMS)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
