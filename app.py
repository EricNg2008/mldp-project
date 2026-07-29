import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="HDB Resale Price Estimator", layout="centered")

# Load the trained model and the feature columns it expects (cached so it loads once)
@st.cache_resource
def load_model():
    return joblib.load("hdb_price_model.pkl"), joblib.load("model_columns.pkl")

model, model_columns = load_model()

# Load the dataset (cached) so I can show graphs that compare the estimate to real sales
@st.cache_data
def load_data():
    return pd.read_csv("ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")

data = load_data()

# Dropdown options (the exact values the model was trained on)
TOWNS = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG',
         'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
         'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS',
         'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
         'TOA PAYOH', 'WOODLANDS', 'YISHUN']

# 1 ROOM left out: only 87 were ever resold, so the model cannot price them reliably
FLAT_TYPES = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']

FLAT_MODELS = ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved',
               'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2',
               'Multi Generation', 'New Generation', 'Premium Apartment', 'Premium Apartment Loft',
               'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']

STOREY_RANGES = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18',
                 '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
                 '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']

# Typical floor area per flat type, used as a smart default so the two stay consistent
TYPICAL_SQM = {'2 ROOM': 47, '3 ROOM': 67, '4 ROOM': 93,
               '5 ROOM': 117, 'EXECUTIVE': 146, 'MULTI-GENERATION': 164}

st.title("HDB Resale Price Estimator")
st.write("Enter a flat's details to get an estimated fair resale price. "
         "Use it as a benchmark to check whether a seller's asking price looks reasonable.")

st.subheader("Flat details")

col1, col2 = st.columns(2)
with col1:
    town = st.selectbox("Town", TOWNS)
    flat_type = st.selectbox("Flat type", FLAT_TYPES, index=FLAT_TYPES.index("4 ROOM"))
    flat_model = st.selectbox("Flat model", FLAT_MODELS, index=FLAT_MODELS.index("Model A"))
with col2:
    storey_range = st.selectbox("Storey (which floors)", STOREY_RANGES, index=3)
    # The floor area slider starts at the average size for the chosen flat type
    # (from the TYPICAL_SQM table above). So when the user picks a different flat
    # type, the sqm automatically jumps to that type's average, so the two inputs
    # always make sense together (a 3 ROOM flat is not left sitting at 146 sqm).
    # The user can still drag the slider to their flat's exact size if they know it.
    floor_area = st.slider(
        "Floor area (sqm)", 30, 200, TYPICAL_SQM[flat_type],
        help="Starts at the average floor area for the flat type you picked. "
             "Drag it if your flat is bigger or smaller.")
    remaining_lease = st.slider("Remaining lease (years)", 40, 99, 90)

year = st.slider("Year of sale", 2017, 2026, 2026)

if flat_type == "MULTI-GENERATION":
    st.info("Note: multi-generation flats are rare in the data, so this estimate is less reliable.")

if st.button("Estimate price"):
    try:
        # storey range "10 TO 12" -> middle floor 11, like in training
        low, high = storey_range.split(" TO ")
        storey_mid = (int(low) + int(high)) // 2

        input_df = pd.DataFrame([{
            "floor_area_sqm": floor_area,
            "remaining_lease_years": remaining_lease,
            "storey_mid": storey_mid,
            "txn_year": year,
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
        }])

        # one-hot encode and line the columns up with the model's training columns
        input_encoded = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)

        price = model.predict(input_encoded)[0]
        st.success(f"Estimated fair resale price: ${price:,.0f}")
        st.caption("This is an estimate. On unseen test data the model is off by about "
                   "$29,000 on average, so treat it as a ballpark rather than an exact valuation.")

        # ---------- Context graphs ----------
        st.subheader("How this estimate compares")

        # Graph 1: the estimate against actual past sales of the same flat type in the same town
        similar = data[(data['town'] == town) & (data['flat_type'] == flat_type)]
        if len(similar) >= 10:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(similar['resale_price'], bins=30, color='#7fb3d5')
            ax.axvline(price, color='red', linestyle='--', linewidth=2, label='Your estimate')
            ax.set_xlabel('Resale Price (SGD)')
            ax.set_ylabel('Number of past sales')
            ax.set_title(f'Your estimate vs past {flat_type} sales in {town}')
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("Not enough past sales of this exact flat type in this town to plot.")

        # Graph 2: how prices in this town have moved over the years
        town_data = data[data['town'] == town].copy()
        town_data['year'] = town_data['month'].str[:4]
        st.write(f"Median resale price by year in {town}:")
        st.line_chart(town_data.groupby('year')['resale_price'].median())

    except Exception as error:
        st.error(f"Sorry, something went wrong while estimating the price: {error}")
