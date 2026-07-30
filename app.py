import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="HDB Resale Price Estimator", page_icon="🏙️", layout="wide")

# ---------- Styling: keep Streamlit's default background, just add a single accent colour ----------
ACCENT = "#0e6ba8"
st.markdown(
    f"""
    <style>
    /* hide default streamlit chrome for a cleaner site look */
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1150px;}}

    /* hero header (single accent colour) */
    .hero {{background: {ACCENT}; padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.3rem;}}
    .hero h1 {{color: #ffffff; margin: 0; font-size: 1.9rem; font-weight: 700;}}
    .hero p {{color: #dceefb; margin: 0.35rem 0 0; font-size: 1.02rem;}}

    /* estimated price card (same single accent colour) */
    .price-card {{background: {ACCENT}; border-radius: 16px; padding: 1.4rem; text-align: center; color: #ffffff;}}
    .price-card .label {{font-size: 0.9rem; letter-spacing: 0.6px; opacity: 0.92; text-transform: uppercase;}}
    .price-card .value {{font-size: 2.6rem; font-weight: 800; margin-top: 0.25rem; line-height: 1;}}
    .price-card .sub {{font-size: 0.82rem; opacity: 0.9; margin-top: 0.5rem;}}

    /* section titles inherit the theme text colour so they work on light or dark */
    .section-title {{font-weight: 700; margin: 1.2rem 0 0.4rem; font-size: 1.05rem;}}
    .foot {{color: #7f8c9a; font-size: 0.8rem; text-align: center; margin-top: 2rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model, feature columns and dataset (cached) ----------
@st.cache_resource
def load_model():
    return joblib.load("hdb_price_model.pkl"), joblib.load("model_columns.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")
    df["year"] = df["month"].str[:4].astype(int)
    return df

model, model_columns = load_model()
data = load_data()

# ---------- Dropdown options (the exact values the model was trained on) ----------
TOWNS = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG',
         'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
         'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS',
         'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
         'TOA PAYOH', 'WOODLANDS', 'YISHUN']

# 1 ROOM left out: only 87 were ever resold, so the model cannot price them reliably
FLAT_TYPES = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']

STOREY_RANGES = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18',
                 '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36',
                 '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']

# Typical floor area per flat type. Used as the floor area default once a flat type is picked,
# so the sqm auto matches the chosen type instead of the user having to guess it.
TYPICAL_SQM = {'2 ROOM': 47, '3 ROOM': 67, '4 ROOM': 93,
               '5 ROOM': 117, 'EXECUTIVE': 146, 'MULTI-GENERATION': 164}

# ---------- Header ----------
st.markdown(
    '<div class="hero"><h1>HDB Resale Price Estimator</h1>'
    '<p>Check a fair resale price for any HDB flat in Singapore before you make an offer.</p></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.25], gap="large")

# ---------- Left: inputs (all start empty) ----------
with left:
    st.markdown('<div class="section-title">Enter the flat details</div>', unsafe_allow_html=True)
    town = st.selectbox("Town", TOWNS, index=None, placeholder="Choose a town")
    flat_type = st.selectbox("Flat type", FLAT_TYPES, index=None, placeholder="Choose a flat type")
    storey_range = st.selectbox("Storey (which floors)", STOREY_RANGES, index=None,
                                placeholder="Choose the storey range")
    # floor area starts at the average size for the chosen flat type (90 before one is picked)
    floor_area = st.slider("Floor area (sqm)", 30, 200, TYPICAL_SQM.get(flat_type, 90))
    st.caption("Floor area auto-sets to the average size for the flat type you pick. "
               "Drag it if you know your flat's exact size.")
    remaining_lease = st.slider("Remaining lease (years)", 40, 99, 90)
    year = st.slider("Year of sale", 2017, 2026, 2026)

# ---------- Right: estimate + comparisons ----------
with right:
    ready = all(v is not None for v in (town, flat_type, storey_range))

    if not ready:
        st.info("Fill in all the flat details on the left to see the estimated fair price "
                "and how it compares to real past sales.")
    else:
        try:
            # storey range "10 TO 12" -> middle floor 11, like in training
            low, high = storey_range.split(" TO ")
            storey_mid = (int(low) + int(high)) // 2

            # comparable past sales: same flat type in the same town
            similar = data[(data["town"] == town) & (data["flat_type"] == flat_type)]

            # flat_model is not asked (buyers rarely know it), so I fill in the most common
            # model for the chosen town and flat type so the model still gets a realistic value.
            flat_model = similar["flat_model"].mode().iloc[0] if len(similar) else "Model A"

            input_df = pd.DataFrame([{
                "floor_area_sqm": floor_area,
                "remaining_lease_years": remaining_lease,
                "storey_mid": storey_mid,
                "txn_year": year,
                "town": town,
                "flat_type": flat_type,
                "flat_model": flat_model,
            }])

            # one hot encode and line the columns up with the model's training columns
            input_encoded = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)
            price = model.predict(input_encoded)[0]

            st.markdown(
                f'<div class="price-card"><div class="label">Estimated fair resale price</div>'
                f'<div class="value">${price:,.0f}</div>'
                f'<div class="sub">Typically within about $29,000 of the real price</div></div>',
                unsafe_allow_html=True,
            )

            # ---------- charts in tabs (no scrolling to switch between them) ----------
            st.markdown('<div class="section-title">How this compares</div>', unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["Vs recent sales", "Price trend"])

            with tab1:
                recent = similar[similar["year"] >= 2023]
                plot_src = recent if len(recent) >= 30 else similar
                if len(plot_src) >= 10:
                    fig, ax = plt.subplots(figsize=(6, 3.2))
                    ax.hist(plot_src["resale_price"], bins=25, color=ACCENT, alpha=0.85)
                    ax.axvline(price, color="#e67e22", linestyle="--", linewidth=2.5, label="Your estimate")
                    ax.set_xlabel("Resale price (SGD)")
                    ax.set_ylabel("Number of sales")
                    ax.set_title(f"Recent {flat_type} sales in {town}")
                    ax.legend()
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    st.pyplot(fig)
                    st.caption("The orange line is your estimate against real recent sales of the same flat type in this town.")
                else:
                    st.write("Not enough past sales of this flat type in this town to plot.")

            with tab2:
                trend = data[data["town"] == town].groupby("year")["resale_price"].median()
                fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                ax2.plot(trend.index, trend.values, marker="o", color=ACCENT, linewidth=2.5)
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Median resale price (SGD)")
                ax2.set_title(f"Median resale price by year in {town}")
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                st.pyplot(fig2)
                st.caption("How resale prices in this town have moved over the years, so you can see if the market is rising.")

        except Exception as error:
            st.error(f"Sorry, something went wrong while estimating the price: {error}")

st.markdown('<div class="foot">Data from data.gov.sg. Estimates are a guide, not an official valuation.</div>',
            unsafe_allow_html=True)
