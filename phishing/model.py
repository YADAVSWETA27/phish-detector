"""Train, evaluate, save and use the classifier."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, explain, vector


def build_matrix(rows):
    X = np.array([vector(u) for u, _ in rows])
    y = np.array([lab for _, lab in rows])
    return X, y


def metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def train(rows, test_size: float = 0.2, seed: int = 42) -> dict:
    """Train a random forest and a simple baseline on the SAME split; return everything useful.

    The test part is never used for fitting, so its scores estimate performance on unseen URLs.
    """
    X, y = build_matrix(rows)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    forest = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1)
    forest.fit(Xtr, ytr)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    baseline.fit(Xtr, ytr)
    importances = sorted(zip(FEATURE_NAMES, forest.feature_importances_), key=lambda t: -t[1])
    return {"model": forest, "forest": metrics(yte, forest.predict(Xte)),
            "baseline": metrics(yte, baseline.predict(Xte)), "importances": importances,
            "n_train": len(ytr), "n_test": len(yte)}


def save(model, path: str) -> None:
    joblib.dump({"model": model, "features": FEATURE_NAMES}, path)


def load(path: str):
    blob = joblib.load(path)
    if blob["features"] != FEATURE_NAMES:
        raise ValueError("This model was trained with different features. Train it again.")
    return blob["model"]


def check(model, url: str, threshold: float = 0.5) -> dict:
    """Score one URL. Returns probability, verdict and the plain-English red flags."""
    p = float(model.predict_proba([vector(url)])[0][1])
    return {"url": url, "phishing_probability": p,
            "verdict": "PHISHING" if p >= threshold else "LEGITIMATE", "reasons": explain(url)}
