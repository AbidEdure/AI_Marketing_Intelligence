# src/preprocess.py
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Column groups — must match the CSV headers exactly
# ---------------------------------------------------------------
NUMERIC_COLS = [
    "age", "annual_income", "days_since_signup", "recency_days",
    "purchase_frequency_12m", "average_order_value", "total_spend_12m",
    "website_visits_30d", "email_opens_30d", "avg_session_minutes",
    "discount_usage_rate", "previous_campaigns",
    "previous_campaign_responses", "campaign_discount_percent",
]

CATEGORICAL_COLS = [
    "gender", "city_tier", "preferred_channel",
    "favorite_category", "membership_tier", "campaign_type",
]

TARGET_COL = "campaign_accepted"
ID_COL = "customer_id"


# ---------------------------------------------------------------
# 1) Load
# ---------------------------------------------------------------
def load_raw(path: str = "data/raw/marketing_intelligence_customers_cleaned.csv") -> pd.DataFrame:
    """Load CSV safely (handles Windows BOM + stray whitespace in headers)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------
# 2) Clean
# ---------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)
    return df


# ---------------------------------------------------------------
# 3) Feature engineering
# ---------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio / aggregate features used by both segmentation & classification."""
    df = df.copy()

    # Ratios
    df["spend_per_order"] = df["total_spend_12m"] / (df["purchase_frequency_12m"] + 1)
    df["orders_per_visit"] = df["purchase_frequency_12m"] / (df["website_visits_30d"] + 1)
    df["email_engagement"] = df["email_opens_30d"] / (df["website_visits_30d"] + 1)
    df["campaign_response_rate"] = df["previous_campaign_responses"] / (df["previous_campaigns"] + 1)
    df["income_per_age"] = df["annual_income"] / (df["age"] + 1)
    df["tenure_years"] = df["days_since_signup"] / 365.0

    # RFM-style quantile scores (guard against tiny datasets)
    if len(df) >= 5:
        df["R_score"] = pd.qcut(
            df["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]
        ).astype(int)
        df["F_score"] = pd.qcut(
            df["purchase_frequency_12m"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        df["M_score"] = pd.qcut(
            df["total_spend_12m"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
    else:
        df["R_score"] = 3
        df["F_score"] = 3
        df["M_score"] = 3

    return df


# ---------------------------------------------------------------
# 4) Classifier frame  (X, y, feature_cols, cat_features)
# ---------------------------------------------------------------
def build_model_frame(df: pd.DataFrame):
    """
    Returns (X, y, feature_cols, categorical_features).

    NOTE: Do NOT try to build the de-duped list by iterating a variable
    that you initialise in the same statement — see git history. This
    version uses a separate `raw_cols` list and a fresh `feature_cols`.
    """
    df = engineer_features(clean(df))

    raw_cols = (
        NUMERIC_COLS
        + [c for c in CATEGORICAL_COLS if c in df.columns]
        + [
            "spend_per_order", "orders_per_visit", "email_engagement",
            "campaign_response_rate", "income_per_age", "tenure_years",
            "R_score", "F_score", "M_score",
        ]
    )

    # De-dupe while preserving order
    seen = set()
    feature_cols = []
    for c in raw_cols:
        if c not in seen and c in df.columns:
            feature_cols.append(c)
            seen.add(c)

    print(f"[preprocess] raw_cols={len(raw_cols)}  kept={len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int) if TARGET_COL in df.columns else None
    cat_features = [c for c in CATEGORICAL_COLS if c in X.columns]
    return X, y, feature_cols, cat_features


# ---------------------------------------------------------------
# 5) Segmentation frame (numeric-only matrix for KMeans)
# ---------------------------------------------------------------
def build_segmentation_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(clean(df))
    cols = [
        "recency_days", "purchase_frequency_12m", "total_spend_12m",
        "average_order_value", "website_visits_30d", "email_opens_30d",
        "discount_usage_rate", "avg_session_minutes",
        "spend_per_order", "campaign_response_rate",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()