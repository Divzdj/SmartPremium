# 💰 SmartPremium: Predicting Insurance Costs with Machine Learning

## Project Overview
SmartPremium is a machine learning project to predict insurance premiums based on customer characteristics and policy details. The project uses **XGBoost** for high-accuracy predictions and provides a **Streamlit web app** for real-time premium estimation.

## Features
- Predicts insurance premiums based on age, income, health, occupation, location, policy type, and more.
- Handles categorical and numerical features with preprocessing pipelines.
- Integrated **MLflow** for experiment tracking and model logging.
- Deployed via **Streamlit** for a user-friendly interface.

## Dataset
- **Download Link:** [Google Drive](https://drive.google.com/drive/folders/1GNSocgMntDHdTVmT2q0p1sE5iZss2h5_?usp=drive_link)  
- **Format:** CSV  
- **Target Variable:** `Premium Amount`  
- **Size:** 2L+ records, 20+ features (categorical, numerical, text)

## ML Approach
- Regression models used: Linear Regression, Decision Tree, Random Forest, **XGBoost (Best)**  
- Evaluation Metrics: RMSE, R², MAE, RMSLE  
- Preprocessing: Missing value handling, scaling, one-hot encoding  

## Web App
- Developed in **Streamlit**  
- Users can input customer details to get **real-time premium predictions**  
- Model pipeline saved as `best_model.pkl`  

## Tech Stack
Python | Pandas | NumPy | Scikit-Learn | XGBoost | MLflow | Streamlit | Git/GitHub

## Project Deliverables
- Jupyter Notebook with code & results  
- ML Pipeline with MLflow integration  
- Trained XGBoost model (`best_model.pkl`)  
- Streamlit Web App code & link
