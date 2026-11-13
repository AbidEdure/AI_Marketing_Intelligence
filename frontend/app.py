# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API = "http://localhost:8000"

st.set_page_config(page_title="AI Marketing Intelligence", layout="centered")
st.title("🎯 AI-Powered Marketing Intelligence")

tab1, tab2 = st.tabs(["🔮 Predict Campaign Acceptance", "👥 Customer Segments"])

# ---------------- TAB 1: Prediction ----------------
with tab1:
    st.subheader("Customer Profile & Campaign Offer")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 18, 100, 32)
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer_not_to_say"])
            city_tier = st.selectbox("City Tier", ["Tier_1", "Tier_2", "Tier_3"])
            annual_income = st.number_input("Annual Income", 0.0, 1_000_000.0, 81093.0)
            days_since_signup = st.number_input("Days Since Signup", 0, 5000, 358)

        with c2:
            recency_days = st.number_input("Recency (days)", 0, 365, 53)
            purchase_frequency = st.number_input("Purchases (12m)", 0, 100, 11)
            average_order_value = st.number_input("Avg Order Value", 0.0, 100000.0, 2352.05)
            total_spend = st.number_input("Total Spend (12m)", 0.0, 500000.0, 24981.0)
            website_visits = st.number_input("Website Visits (30d)", 0, 200, 14)

        with c3:
            email_opens = st.number_input("Email Opens (30d)", 0, 100, 3)
            avg_session = st.number_input("Avg Session (min)", 0.0, 120.0, 14.4)
            discount_rate = st.slider("Discount Usage Rate", 0.0, 1.0, 0.18)
            prev_camp = st.number_input("Previous Campaigns", 0, 50, 3)
            prev_resp = st.number_input("Previous Responses", 0, 50, 0)

        st.markdown("**Campaign Offer**")
        d1, d2, d3 = st.columns(3)
        with d1:
            preferred_channel = st.selectbox("Preferred Channel", ["Email", "SMS", "Social_Media", "Push_Notification"])
        with d2:
            favorite_category = st.selectbox("Favorite Category", ["Electronics", "Fashion", "Home", "Beauty", "Sports"])
            membership_tier = st.selectbox("Membership Tier", ["Basic", "Silver", "Gold", "Platinum"])
        with d3:
            campaign_type = st.selectbox("Campaign Type", ["Discount", "Seasonal", "New_Product", "Loyalty"])
            campaign_discount = st.slider("Campaign Discount %", 0, 50, 13)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        payload = {
            "age": age, "gender": gender, "city_tier": city_tier,
            "annual_income": annual_income, "days_since_signup": days_since_signup,
            "recency_days": recency_days, "purchase_frequency_12m": purchase_frequency,
            "average_order_value": average_order_value, "total_spend_12m": total_spend,
            "website_visits_30d": website_visits, "email_opens_30d": email_opens,
            "avg_session_minutes": avg_session, "discount_usage_rate": discount_rate,
            "preferred_channel": preferred_channel, "favorite_category": favorite_category,
            "membership_tier": membership_tier, "previous_campaigns": prev_camp,
            "previous_campaign_responses": prev_resp, "campaign_type": campaign_type,
            "campaign_discount_percent": campaign_discount,
        }
        try:
            r = requests.post(f"{API}/predict", json=payload, timeout=15)
            r.raise_for_status()
            res = r.json()

            m1, m2, m3 = st.columns(3)
            m1.metric("Accept Probability", f"{res['accept_probability']*100:.1f}%")
            m2.metric("Will Accept?", "✅ Yes" if res["will_accept"] else "❌ No")
            m3.metric("Segment", res["segment_label"])

            st.markdown("**Why (top reasons)**")
            for reason in res["top_reasons"]:
                st.write(f"- {reason}")

        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")

# ---------------- TAB 2: Segments ----------------
with tab2:
    st.subheader("Customer Segments (from KMeans)")

    if st.button("🔄 Reload segments"):
        st.rerun()

    try:
        r = requests.get(f"{API}/segments", timeout=5)
        r.raise_for_status()
        segs = pd.DataFrame(r.json())

        st.dataframe(segs, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(segs, x="segment_label", y="avg_spend",
                         title="Average 12m Spend by Segment", color="segment_label")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(segs, x="avg_recency_days", y="accept_rate",
                              size="size", color="segment_label",
                              title="Accept Rate vs Recency (bubble=size)")
            st.plotly_chart(fig2, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.warning(
            f"⚠️ Cannot reach the API at **{API}**. "
            "Start it with:\n\n"
            "```bash\nuvicorn backend.main:app --reload --port 8000\n```"
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e.response.status_code} — {e.response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")