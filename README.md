Smart Home Energy Consumption Predictor

This project is designed to predict the energy consumption of smart homes based on environmental and usage factors such as temperature, humidity, appliances in use, and time of day. The model uses machine learning to provide accurate predictions, with a user-friendly interface powered by Streamlit.


Features

1. Energy Consumption Prediction: Predicts energy usage based on user input data.
2. Synthetic Data Generation: Generates realistic data for training.
3. Machine Learning Model: Utilizes a Random Forest Regressor for high prediction accuracy.
4. Interactive Interface: Streamlit app provides an easy-to-use interface for input and predictions.

How the Project Works

1. Data Generation:
   - Synthetic data is generated with columns: `temperature`, `humidity`, `appliances_in_use`, `time_of_day`, and `energy_consumption`.
   - A CSV file (`data/energy_data.csv`) is created for storing this synthetic data.

2. Model Training:
   - A Random Forest Regressor is trained on the synthetic data.
   - The trained model is saved as a `.pkl` file (`models/trained_model.pkl`).

3. Prediction:
   - Users provide input values through the Streamlit app (e.g., temperature, humidity, appliances in use, time of day).
   - The app preprocesses the inputs and uses the trained model to predict energy consumption.

4. Interactive Application:
   - Streamlit powers the UI, where users can:
     - Train the model if required.
     - Input data and get predictions.


Technologies Used

- Programming Language: Python
- Libraries:
  - `Pandas` for data manipulation
  - `NumPy` for numerical operations
  - `Scikit-learn` for machine learning
  - `Streamlit` for building the interactive app
- Model: Random Forest Regressor


Algorithms Used

Random Forest Regressor:
  - A robust ensemble machine learning algorithm used for regression tasks.
  - Combines multiple decision trees to improve prediction accuracy and reduce overfitting.



A short demonstration of the application:
- Input values: Temperature = 25°C, Humidity = 60%, Appliances = 3, Time of Day = "evening."
- Predicted Energy Consumption: ~80 kWh.


Future Improvements

1. Enhanced Data: Use real-world data for better accuracy.
2. Additional Features: Incorporate more factors like weather conditions or user habits.
3. Model Optimization: Experiment with other algorithms (e.g., Gradient Boosting, Neural Networks).
4. Deployment: Deploy the app on a cloud platform for public access.



