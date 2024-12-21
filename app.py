 import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import streamlit as st

# Ensure necessary directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Function to generate synthetic data
def generate_data():
    np.random.seed(42)
    data = {
        "temperature": np.random.uniform(15, 30, 1000),
        "humidity": np.random.uniform(30, 90, 1000),
        "appliances_in_use": np.random.randint(1, 10, 1000),
        "time_of_day": np.random.choice(["morning", "afternoon", "evening", "night"], 1000)
    }
    data["energy_consumption"] = (
        0.5 * data["temperature"] +
        0.3 * data["humidity"] +
        10 * data["appliances_in_use"] +
        [20 if tod == "evening" else 10 for tod in data["time_of_day"]] +
        np.random.normal(0, 5, 1000)
    )
    df = pd.DataFrame(data)
    df.to_csv("data/energy_data.csv", index=False)
    print("Synthetic data generated and saved to 'data/energy_data.csv'.")

# Function to preprocess data with consistent features
def preprocess_data(df, expected_columns=None):
    """
    Preprocess the data by encoding categorical variables and aligning columns.
    """
    # Add a placeholder for time_of_day if missing
    if "time_of_day" not in df.columns and expected_columns:
        df["time_of_day"] = "unknown"  # Dummy value to avoid KeyError during preprocessing
    
    # Create dummy variables for 'time_of_day'
    df = pd.get_dummies(df, columns=["time_of_day"], drop_first=True)
    
    # Ensure all expected columns are present
    if expected_columns:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing column with default value 0
        
        # Reorder columns to match the expected order
        df = df[expected_columns]
    
    return df

# Function to train the model
def train_model():
    df = pd.read_csv("data/energy_data.csv")
    df = preprocess_data(df)
    X = df.drop("energy_consumption", axis=1)
    y = df["energy_consumption"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    feature_names = X.columns.tolist()
    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump((model, feature_names), f)
    print("Model trained and saved to 'models/trained_model.pkl'.")

# Function to predict energy consumption
def predict_energy(model, feature_names, input_data):
    """
    Predict energy consumption using the trained model.
    """
    # Add 'time_of_day' column for preprocessing
    input_data["time_of_day"] = input_data.pop("time_of_day", "unknown")
    
    # Convert input data to DataFrame and preprocess
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_data(input_df, expected_columns=feature_names)
    
    # Predict using the model
    return model.predict(input_df)[0]

# Streamlit app
def main():
    st.title("Smart Home Energy Consumption Predictor")
    
    # Sidebar for user input
    st.sidebar.header("Input Features")
    temperature = st.sidebar.slider("Temperature (°C)", 15, 50, 22)
    humidity = st.sidebar.slider("Humidity (%)", 30, 90, 50)
    appliances_in_use = st.sidebar.slider("Appliances in Use", 1, 5, 3)
    time_of_day = st.sidebar.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
    
    # Input data dictionary
    input_data = {
        "temperature": temperature,
        "humidity": humidity,
        "appliances_in_use": appliances_in_use,
        "time_of_day": time_of_day  # Include raw time_of_day for preprocessing
    }
    
    # Display input data
    st.write("### Input Data", pd.DataFrame([input_data]))

    # Predict energy consumption
    if st.button("Predict Energy Consumption"):
        try:
            with open("models/trained_model.pkl", "rb") as f:
                model, feature_names = pickle.load(f)
            prediction = predict_energy(model, feature_names, input_data)
            st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")
        except FileNotFoundError:
            st.error("Model not found. Please train the model first.")
    
    # Option to train the model
    if st.button("Train Model"):
        generate_data()
        train_model()
        st.success("Model trained successfully!")

if __name__ == "__main__":
    main()
