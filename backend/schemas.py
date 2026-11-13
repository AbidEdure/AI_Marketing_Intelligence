# backend/schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

Gender = Literal["Male", "Female", "Other", "Prefer_not_to_say"]
CityTier = Literal["Tier_1", "Tier_2", "Tier_3"]
Channel = Literal["Email", "SMS", "Social_Media", "Push_Notification"]
Category = Literal["Electronics", "Fashion", "Home", "Beauty", "Sports"]
Membership = Literal["Basic", "Silver", "Gold", "Platinum"]
CampaignType = Literal["Discount", "Seasonal", "New_Product", "Loyalty"]


class CustomerFeatures(BaseModel):
    # identity / demographics
    age: int = Field(..., ge=18, le=100)
    gender: Gender
    city_tier: CityTier
    annual_income: float = Field(..., gt=0)

    # behavioral history
    days_since_signup: int = Field(..., ge=0)
    recency_days: int = Field(..., ge=0)
    purchase_frequency_12m: int = Field(..., ge=0)
    average_order_value: float = Field(..., ge=0)
    total_spend_12m: float = Field(..., ge=0)
    website_visits_30d: int = Field(..., ge=0)
    email_opens_30d: int = Field(..., ge=0)
    avg_session_minutes: float = Field(..., ge=0)
    discount_usage_rate: float = Field(..., ge=0, le=1)
    preferred_channel: Channel
    favorite_category: Category
    membership_tier: Membership
    previous_campaigns: int = Field(..., ge=0)
    previous_campaign_responses: int = Field(..., ge=0)

    # campaign offered
    campaign_type: CampaignType
    campaign_discount_percent: float = Field(..., ge=0, le=100)


class PredictionResponse(BaseModel):
    accept_probability: float
    will_accept: bool
    segment_id: int
    segment_label: str
    top_reasons: list[str]


class SegmentInfo(BaseModel):
    segment_id: int
    segment_label: str
    size: int
    avg_spend: float
    avg_recency_days: float
    accept_rate: float