# src/train.py
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"   # silence Git warnings BEFORE mlflow import

import json
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
)
from xgboost import XGBClassifier

from src.preprocess import (
    load_raw, build_model_frame, build_segmentation_frame,
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Trusted types so MLflow/skops will serialize XGBoost inside the Pipeline
SKOPS_TRUSTED_TYPES = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def _log_sklearn_model(model, name: str):
    """Helper: logs a sklearn Pipeline with XGBoost inside, safely."""
    try:
        mlflow.sklearn.log_model(
            model,
            name=name,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    except TypeError:
        # Older MLflow that doesn't accept `name=` or `skops_trusted_types=`
        mlflow.sklearn.log_model(
            model,
            artifact_path=name,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )


# ---------------------------------------------------------------
# 1) SEGMENTATION (KMeans)
# ---------------------------------------------------------------
def train_segmenter(n_clusters: int = 5):
    df = load_raw()
    seg_X = build_segmentation_frame(df)
    print(f"[debug] segmentation matrix shape = {seg_X.shape}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
    ])
    pipe.fit(seg_X)

    labels = pipe.predict(seg_X)
    X_scaled = pipe.named_steps["scaler"].transform(seg_X)
    sil = silhouette_score(X_scaled, labels)

    mlflow.set_experiment("marketing_segmentation")
    with mlflow.start_run(run_name="kmeans_segmentation"):
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metric("silhouette", sil)
        _log_sklearn_model(pipe, "kmeans_pipeline")

    joblib.dump(pipe, f"{MODELS_DIR}/segmenter.pkl")
    print(f"[segmentation] silhouette={sil:.4f} saved to models/segmenter.pkl")
    return pipe, sil


# ---------------------------------------------------------------
# 2) CLASSIFICATION (XGBoost)
# ---------------------------------------------------------------
def train_classifier():
    df = load_raw()
    X, y, feature_cols, cat_features = build_model_frame(df)

    print(f"[debug] X.shape          = {X.shape}")
    print(f"[debug] feature_cols ({len(feature_cols)}) = {feature_cols}")
    print(f"[debug] cat_features ({len(cat_features)}) = {cat_features}")

    cat_features = [c for c in cat_features if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    print(f"[debug] num_features ({len(num_features)}) = {num_features}")
    print(f"[debug] cat_features (filtered, {len(cat_features)}) = {cat_features}")

    if len(num_features) == 0 and len(cat_features) == 0:
        raise RuntimeError(
            "No features available for training. "
            "Check that CSV headers match NUMERIC_COLS/CATEGORICAL_COLS in preprocess.py."
        )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[debug] train shape={X_tr.shape}, test shape={X_te.shape}")
    print(f"[debug] positive rate -> train={y_tr.mean():.3f}, test={y_te.mean():.3f}")

    X_tr_t = pre.fit_transform(X_tr)
    print(f"[debug] After ColumnTransformer: shape={X_tr_t.shape}, dtype={X_tr_t.dtype}")
    if X_tr_t.shape[1] == 0:
        raise RuntimeError("ColumnTransformer produced 0 features — nothing to train on.")

    params = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    clf_pipe = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(**params)),
    ])

    mlflow.set_experiment("campaign_acceptance")
    with mlflow.start_run(run_name="xgboost_baseline") as run:
        clf_pipe.fit(X_tr, y_tr)

        proba = clf_pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "roc_auc":  roc_auc_score(y_te, proba),
            "accuracy": accuracy_score(y_te, pred),
            "f1":       f1_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall":   recall_score(y_te, pred),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {"feature_cols": feature_cols, "cat_features": cat_features},
            "features.json",
        )
        _log_sklearn_model(clf_pipe, "clf_pipeline")

        joblib.dump(clf_pipe, f"{MODELS_DIR}/classifier.pkl")
        joblib.dump(
            {"feature_cols": feature_cols, "cat_features": cat_features},
            f"{MODELS_DIR}/features.pkl",
        )

        print(f"[classifier] AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")
        print(f"[classifier] confusion_matrix =\n{confusion_matrix(y_te, pred)}")
        print(f"[classifier] saved to models/classifier.pkl (run_id={run.info.run_id})")

    return clf_pipe, metrics


# ---------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------
if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    train_segmenter(n_clusters=5)
    train_classifier()