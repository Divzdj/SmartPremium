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
education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])
occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
location = st.selectbox("Location", ["Urban", "Suburban"])
policy_type = st.selectbox("Policy Type", ["Comprehensive", "Premium"])
smoking_status = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])
exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Occasionally", "Never"])
property_type = st.selectbox("Property Type", ["Condo", "House"])

# --- Columns your model expects ---
expected_columns = [
    'id','Age','Gender','Annual Income','Number of Dependents','Education Level','Health Score',
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

# Map categorical inputs to one-hot encoded columns
input_df.loc[0, f'Marital Status_{marital_status}'] = 1
if occupation == "Self-Employed":
    input_df.loc[0, 'Occupation_Self-Employed'] = 1
elif occupation == "Unemployed":
    input_df.loc[0, 'Occupation_Unemployed'] = 1
# If "Employed", leave both zeros (matches training encoding)

input_df.loc[0, f'Location_{location}'] = 1
input_df.loc[0, f'Property Type_{property_type}'] = 1
input_df.loc[0, f'Policy Type_{policy_type}'] = 1

# Encode Gender, Smoking Status, Exercise Frequency as integers if needed
gender_map = {"Male": 0, "Female": 1, "Other": 2}
smoking_map = {"Non-Smoker": 0, "Smoker": 1}
exercise_map = {"Daily": 0, "Weekly": 1, "Occasionally": 2, "Never": 3}

input_df.loc[0, 'Gender'] = gender_map[gender]
input_df.loc[0, 'Smoking Status'] = smoking_map[smoking_status]
input_df.loc[0, 'Exercise Frequency'] = exercise_map[exercise_frequency]

# --- Predict ---
if st.button("Predict Premium"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Insurance Premium: ${prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
