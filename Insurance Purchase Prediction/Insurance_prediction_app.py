import streamlit as st 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle

# Load trained model
filepath = r"C:\Users\CHARLES\Desktop\Data science work\Insurance project\my_model.pkl"
model = pickle.load(open(filepath, 'rb'))

st.write('''# Insurance Purchase Prediction App
The following model predicts who is interested in vehicle insurance from health insurance owners.''')

st.sidebar.header('Input Feature Selection')

# User input function
def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 10, 100, 30)
    driving_license = st.sidebar.selectbox('Driving License', [0, 1])
    region_code = st.sidebar.number_input('Region Code', min_value=0.0, max_value=70.0)
    previously_insured = st.sidebar.selectbox('Previously Insured', [0, 1])
    vehicle_age = st.sidebar.selectbox('Vehicle Age', ['<1 Year', '1-2 Year', '> 2 Years'])
    vehicle_damage = st.sidebar.selectbox('Vehicle Damage', ['Yes', 'No'])
    annual_premium = st.sidebar.number_input('Annual Premium', value=0.0)
    policy_sales = st.sidebar.number_input('Policy Sales', value=0.0)
    vintage = st.sidebar.slider('Vintage', 0, 300, 100)

    data = {
        'Gender': gender,
        'Age': age,
        'Driving_License': driving_license,
        'Region_Code': region_code,
        'Previously_Insured': previously_insured,
        'Annual_Premium': annual_premium,
        'Policy_Sales_Channel': policy_sales,
        'Vintage': vintage,
        'Vehicle_Age': vehicle_age,
        'Vehicle_Damage': vehicle_damage,
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get input
input_df = user_input_features()

# Encode categorical variables
def preprocess_input(df):
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
    return df

# Preprocess input
processed_input = preprocess_input(input_df)

# Predict
prediction = model.predict(processed_input)
prediction_proba = model.predict_proba(processed_input)

# Display output
st.subheader('Prediction')
st.write('✅ Will Buy Insurance' if prediction[0] == 1 else '❌ Will Not Buy Insurance')

st.subheader('Prediction Probability')
st.write(f"Probability of buying: {prediction_proba[0][1]*100:.2f}%")

