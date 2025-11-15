# SmartPremium: Predicting Insurance Costs with Machine Learning

## Project Overview
SmartPremium is a machine learning project to predict insurance premiums based on customer characteristics and policy details.  
The project uses **XGBoost** for high-accuracy predictions and includes a **Streamlit web app** for real-time premium estimation.

---

## Features
* Predicts insurance premiums based on customer demographics and policy attributes  
* Handles categorical and numerical features using preprocessing pipelines  
* Integrated **MLflow** for experiment tracking  
* Streamlit web app for real-time predictions  

---

## Dataset
* **Download Link:** https://drive.google.com/drive/folders/1GNSocgMntDHdTVmT2q0p1sE5iZss2h5_?usp=drive_link  
* **Format:** CSV  
* **Target Variable:** Premium Amount  
* **Size:** 2L+ records, 20+ features  

---

## ML Approach

### Models Used
* Linear Regression  
* Decision Tree Regressor  
* Random Forest Regressor  
* **XGBoost Regressor (Best Model)**  

### Evaluation Metrics
* RMSE  
* R² Score  
* MAE  
* RMSLE  

### Preprocessing
* Missing value handling  
* Scaling  
* One-hot encoding  
* Hyperparameter tuning  

---

## Web App
* Built using **Streamlit**  
* User inputs customer details and receives real-time premium predictions  
* Model pipeline saved as `best_model.pkl`  

---

## Tech Stack
* Python  
* Pandas  
* NumPy  
* Scikit-Learn  
* XGBoost  
* MLflow  
* Streamlit  
* Git/GitHub  

---

## Project Deliverables
* Jupyter Notebook with complete code and results  
* ML pipeline with MLflow integration  
* Trained XGBoost model (`best_model.pkl`)  
* Streamlit web app code  
