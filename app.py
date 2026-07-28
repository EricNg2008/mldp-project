import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(page_title="HDB Resale Price Estimator", page_icon="🏠", layout="centered")

# ---------- Load the trained model and the columns it expects ----------
# cache_resource loads these once, so the app stays fast between clicks.
@st.cache_resource
def load_model():
    model = joblib.load("hdb_price_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, model_columns = load_model()

# ---------- Dropdown options (the exact values the model was trained on) ----------
TOWNS = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG',
         'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
         'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS',
         'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
         'TOA PAYOH', 'WOODLANDS', 'YISHUN']

FLAT_TYPES = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']

FLAT_MODELS = ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved',
               'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2',
               'Multi Generation', 'New Generation', 'Premium Apartment', 'Premium Apartment Loft',
               'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']

STOREY_RANGES = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18',
                 '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
                 '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']

# ---------- Header ----------
st.title("🏠 HDB Resale Price Estimator")
st.write(
    "Enter a flat's details to get an estimated fair resale price. "
    "Use it as a benchmark to check whether a seller's asking price looks reasonable."
)

# ---------- User inputs ----------
# We ask for details the way a buyer already knows them (e.g. the storey range and
# the lease years left), and convert them into model inputs behind the scenes.
st.subheader("Flat details")

col1, col2 = st.columns(2)
with col1:
    town = st.selectbox("Town", TOWNS)
    flat_type = st.selectbox("Flat type", FLAT_TYPES, index=3)                 # default 4 ROOM
    flat_model = st.selectbox("Flat model", FLAT_MODELS,
                              index=FLAT_MODELS.index("Model A"))               # default Model A
with col2:
    storey_range = st.selectbox("Storey (which floors)", STOREY_RANGES, index=3)  # default 10 TO 12
    floor_area = st.slider("Floor area (sqm)", 30, 200, 90)
    remaining_lease = st.slider("Remaining lease (years)", 40, 99, 90)

year = st.slider("Year of sale", 2017, 2026, 2026)

# ---------- Predict ----------
if st.button("Estimate price"):
    try:
        # Turn the storey range ("10 TO 12") into its middle floor, exactly like in training
        low, high = storey_range.split(" TO ")
        storey_mid = (int(low) + int(high)) // 2

        # Build a one-row DataFrame with the same raw columns as the training data
        input_df = pd.DataFrame([{
            "floor_area_sqm": floor_area,
            "remaining_lease_years": remaining_lease,
            "storey_mid": storey_mid,
            "txn_year": year,
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
        }])

        # One-hot encode and line the columns up with the training columns.
        # reindex adds any missing dummy columns as 0 so the shape matches the model.
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Predict the price
        price = model.predict(input_encoded)[0]

        st.success(f"Estimated fair resale price: ${price:,.0f}")
        st.caption(
            "This is an estimate. On unseen test data the model is off by about "
            "$29,000 on average, so treat it as a ballpark rather than an exact valuation."
        )
    except Exception as error:
        # User-facing error message if anything goes wrong
        st.error(f"Sorry, something went wrong while estimating the price: {error}")
