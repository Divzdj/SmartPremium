import streamlit as st

import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("notebooks/best_model.pkl")

st.title("Insurance Premium Prediction")
st.write("Enter customer details to predict the insurance premium:")

# --- User Inputs ---
age = st.number_input("Age", 18, 100, 30)
annual_income = st.number_input("Annual Income", 10000, 1000000, 50000)
num_dependents = st.number_input("Number of Dependents", 0, 10, 0)
health_score = st.slider("Health Score", 0, 100, 50)
previous_claims = st.number_input("Previous Claims", 0, 50, 0)
vehicle_age = st.number_input("Vehicle Age", 0, 30, 5)
credit_score = st.number_input("Credit Score", 300, 850, 650)
insurance_duration = st.number_input("Insurance Duration (Years)", 1, 50, 1)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
location = st.selectbox("Location", ["Urban", "Suburban"])
policy_type = st.selectbox("Policy Type", ["Comprehensive", "Premium"])
smoking_status = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])
exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
property_type = st.selectbox("Property Type", ["Condo", "House"])

# --- Columns your model expects ---
expected_columns = [
    'Age','Gender','Annual Income','Number of Dependents','Education Level','Health Score',
    'Previous Claims','Vehicle Age','Credit Score','Insurance Duration','Smoking Status',
    'Exercise Frequency','Marital Status_Married','Marital Status_Single',
    'Occupation_Self-Employed','Occupation_Unemployed','Location_Suburban','Location_Urban',
    'Property Type_Condo','Property Type_House','Policy Type_Comprehensive','Policy Type_Premium'
]

# Initialize input dataframe with zeros
input_df = pd.DataFrame(0, index=[0], columns=expected_columns)

# Fill numeric values
input_df.loc[0, 'Age'] = age
input_df.loc[0, 'Annual Income'] = annual_income
input_df.loc[0, 'Number of Dependents'] = num_dependents
input_df.loc[0, 'Health Score'] = health_score
input_df.loc[0, 'Previous Claims'] = previous_claims
input_df.loc[0, 'Vehicle Age'] = vehicle_age
input_df.loc[0, 'Credit Score'] = credit_score
input_df.loc[0, 'Insurance Duration'] = insurance_duration

# Encode categorical values
gender_map = {"Male": 0, "Female": 1, "Other": 2}
smoking_map = {"Non-Smoker": 0, "Smoker": 1}
exercise_map = {"Rarely":1, "Monthly":2, "Weekly":3, "Daily":4}
education_map = {"High School":1, "Bachelor's":2, "Master's":3, "PhD":4}

input_df.loc[0, 'Gender'] = gender_map[gender]
input_df.loc[0, 'Smoking Status'] = smoking_map[smoking_status]
input_df.loc[0, 'Exercise Frequency'] = exercise_map[exercise_frequency]
input_df.loc[0, 'Education Level'] = education_map[education_level]

# One-hot encode remaining categorical columns safely
one_hot_cols = {
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Location': location,
    'Property Type': property_type,
    'Policy Type': policy_type
}

for prefix, value in one_hot_cols.items():
    col_name = f"{prefix}_{value}"
    if col_name in input_df.columns:
        input_df.loc[0, col_name] = 1

# --- Predict ---
if st.button("Predict Premium"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Insurance Premium: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
