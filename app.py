import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\munni\OneDrive\Desktop\fraud detection\creditcard.zip")
    return df

# Function to train the model
def train_model(df):
    # Prepare the data
    X = df[['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5']]
    y = df['Class']

    # Scale features
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=12)
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Function to make predictions
def predict_fraud(model, features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return 'Fraud' if prediction[0] == 1 else 'Not Fraud'

# Streamlit UI
st.title("Fraud Detection System")

st.write("""
This application predicts whether a transaction is fraudulent based on input features.
""")

# Load data
df = load_data()

# Train model on button click
if st.button("Train Model"):
    model, X_test, y_test = train_model(df)
    st.success("Model trained successfully!")

# Input fields for simplified features
time = st.number_input("Transaction Time", min_value=0)
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
v1 = st.slider("V1", -50.0, 50.0, 0.0)
v2 = st.slider("V2", -50.0, 50.0, 0.0)
v3 = st.slider("V3", -50.0, 50.0, 0.0)
v4 = st.slider("V4", -50.0, 50.0, 0.0)
v5 = st.slider("V5", -50.0, 50.0, 0.0)

# When the button is clicked, make a prediction
if st.button("Predict"):
    if 'model' in locals():
        features = [time, amount, v1, v2, v3, v4, v5]
        result = predict_fraud(model, features)
        st.write(f"Prediction: {result}")
    else:
        st.error("Please train the model first by clicking the 'Train Model' button.")
import time
import logging

logging.basicConfig(level=logging.INFO)

def train_model(df):
    logging.info("Training started")
    start_time = time.time()

    # Data preparation
    logging.info("Preparing data...")
    X = df[['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5']]
    y = df['Class']

    # Scaling
    logging.info("Scaling data...")
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    # Splitting data
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    logging.info("Training model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=12)
    model.fit(X_train, y_train)

    logging.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    return model, X_test, y_test
with st.spinner("Training the model..."):
    model, X_test, y_test = train_model(df)
st.success("Model trained successfully!")
