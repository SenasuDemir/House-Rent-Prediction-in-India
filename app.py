from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
import pandas as pd
import joblib

# Load data and model
df = pd.read_csv('House_Rent_Dataset.csv')
df['Extracted Floor'] = df['Floor'].str.extract(r'^(\d{1,2}|[A-Za-z]+)', expand=False)

def map_floors(floor):
    if floor.startswith('Ground'):
        return 0
    elif floor.startswith('Upper'):
        return 0
    elif floor.startswith('Lower'):
        return -1
    else:
        return floor

df['Extracted Floor'] = df['Extracted Floor'].apply(map_floors)

# Load the trained model
model = joblib.load('best_regression_model.pkl')

# Define the preprocessor and pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["BHK", "Size", 'Bathroom', 'Extracted Floor']),
        ("cat", OneHotEncoder(), ["Area Type", "Area Locality", "City", 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])
    ]
)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

# Fit the pipeline on the data (optional, if you want to refit with the data)
pipeline.fit(df[['BHK', 'Size', 'Area Type', 'Area Locality', 'City',
                 'Furnishing Status', 'Tenant Preferred', 'Bathroom', 'Point of Contact', 'Extracted Floor']], df["Rent"])

# Function to predict the price
def price_prediction(bhk, size, area_type, area_locality, city, furnishing_status, tenant_preferred, bathroom, point_of_contact, floor):
    input_data = pd.DataFrame({
        "BHK": [bhk],
        "Size": [size],
        "Area Type": [area_type],
        "Area Locality": [area_locality],
        "City": [city],
        "Furnishing Status": [furnishing_status],
        "Tenant Preferred": [tenant_preferred],
        "Bathroom": [bathroom],
        "Point of Contact": [point_of_contact],
        "Extracted Floor": [floor]
    })
    prediction = pipeline.predict(input_data)[0]
    return prediction

# Main function to render the Streamlit app
def main():
    st.set_page_config(page_title="House Rent Prediction in India", layout="wide")
    
    # App title and description
    st.title("🏠 House Rent Prediction in India")
    st.markdown("""
    **Enter the house features** below to predict the rent. 
    Adjust the inputs to see how different characteristics affect the rent.
    """)
    
    # Add custom CSS for styling
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-size: 18px;
        border-radius: 5px;
        padding: 10px 20px;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .stText {
        font-size: 16px;
        color: #333;
    }
    .stTitle {
        color: #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Side bar inputs (Better structure and user-friendly)
    st.sidebar.header("Enter House Details")
    
    city = st.sidebar.selectbox("City", df["City"].unique())
    bhk = st.sidebar.number_input("Number of Bedrooms, Hall, Kitchen", int(df["BHK"].min()), int(df["BHK"].max()))
    size = st.sidebar.number_input("Size of the House in Square Feet", min_value=float(df["Size"].min()), max_value=float(df["Size"].max()), step=10.0)
    bathroom = st.sidebar.number_input("Bathroom Number", 0, step=1)
    floor = st.sidebar.number_input("Extracted Floor", -1, step=1)

    # Dynamic options based on city selection
    area_type = st.sidebar.selectbox("Area Type", df[df['City'] == city]['Area Type'].unique())
    area_locality = st.sidebar.selectbox("Area Locality", df[df['City'] == city]['Area Locality'].unique())
    tenant_preferred = st.sidebar.selectbox("Tenant Preferred", df["Tenant Preferred"].unique())
    furnishing_status = st.sidebar.selectbox("Furnishing Status", df["Furnishing Status"].unique())
    point_of_contact = st.sidebar.selectbox("Point of Contact", df["Point of Contact"].unique())
    
    # Prediction button
    if st.sidebar.button("Predict Rent"):
        price = price_prediction(bhk, size, area_type, area_locality, city, furnishing_status, tenant_preferred, bathroom, point_of_contact, floor)
        price = float(price)
        
        # Display the result with enhanced visualization
        st.subheader("Predicted Rent: 💲 **${:,.2f}**".format(price))
        st.markdown("""
        This is the estimated price based on the characteristics you provided. 
        Please note that the actual market rent may vary.
        """)

if __name__ == "__main__":
    main()
