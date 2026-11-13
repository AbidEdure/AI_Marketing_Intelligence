# backend/main.py
import sys, os, joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import engineer_features, build_model_frame, load_raw, build_segmentation_frame
from backend.schemas import CustomerFeatures, PredictionResponse, SegmentInfo

MODELS_DIR = "models"

# ---------- Load artifacts once ------------------------------------
try:
    clf = joblib.load(f"{MODELS_DIR}/classifier.pkl")
    seg = joblib.load(f"{MODELS_DIR}/segmenter.pkl")
    feat_spec = joblib.load(f"{MODELS_DIR}/features.pkl")
except FileNotFoundError as e:
    raise RuntimeError(
        "Model artifacts missing. Run `python -m src.train` first."
    ) from e

FEATURE_COLS = feat_spec["feature_cols"]
CAT_FEATURES = feat_spec["cat_features"]

app = FastAPI(title="AI Marketing Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Human-readable labels; refine after inspecting cluster centers
SEGMENT_LABELS = {
    0: "High-Value Loyalists",
    1: "At-Risk Big Spenders",
    2: "Price-Sensitive Browsers",
    3: "Emerging New Customers",
    4: "Engaged Mid-Tier",
}


def _to_frame(payload: CustomerFeatures) -> pd.DataFrame:
    row = payload.model_dump()
    row["customer_id"] = "API_CALL"
    row["campaign_accepted"] = 0  # placeholder; not used as feature
    return pd.DataFrame([row])


def _top_reasons(row: pd.Series) -> list[str]:
    """Lightweight heuristic explanation (swap for SHAP later)."""
    reasons = []

    prev_resp = row.get("previous_campaign_responses", 0)
    if prev_resp is not None and prev_resp > 0:
        reasons.append(f"Responded to {int(prev_resp)} past campaigns")

    disc_use = row.get("discount_usage_rate", 0)
    if disc_use is not None and disc_use > 0.5:
        reasons.append("Heavy discount user — likely coupon-driven")

    recency = row.get("recency_days", 999)
    if recency is not None and recency < 30:
        reasons.append("Recently active (recency < 30 days)")

    camp_disc = row.get("campaign_discount_percent", 0)
    if camp_disc is not None and camp_disc >= 20:
        reasons.append(f"Attractive {camp_disc}% discount offered")

    if not reasons:
        reasons.append("Baseline model inference — no standout behavior")
    return reasons[:3]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures):
    try:
        df_raw = _to_frame(payload)

        # Classifier path — reuse the exact training feature engineering
        X, _, _, _ = build_model_frame(df_raw)
        X = X.reindex(columns=FEATURE_COLS)

        proba = float(clf.predict_proba(X)[0, 1])

        # Segmenter path
        seg_X = build_segmentation_frame(df_raw)
        seg_id = int(seg.predict(seg_X)[0])

        return PredictionResponse(
            accept_probability=round(proba, 4),
            will_accept=proba >= 0.5,
            segment_id=seg_id,
            segment_label=SEGMENT_LABELS.get(seg_id, f"Segment {seg_id}"),
            top_reasons=_top_reasons(X.iloc[0]),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/segments", response_model=list[SegmentInfo])
def segments():
    df = load_raw()
    seg_X = build_segmentation_frame(df)
    df["segment_id"] = seg.predict(seg_X)

    out = []
    for sid, g in df.groupby("segment_id"):
        out.append(SegmentInfo(
            segment_id=int(sid),
            segment_label=SEGMENT_LABELS.get(int(sid), f"Segment {sid}"),
            size=int(len(g)),
            avg_spend=round(float(g["total_spend_12m"].mean()), 2),
            avg_recency_days=round(float(g["recency_days"].mean()), 1),
            accept_rate=round(float(g["campaign_accepted"].mean()), 3),
        ))
    return sorted(out, key=lambda s: -s.size)