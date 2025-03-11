# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Preprocess the data (same as in train.py)
def preprocess_data(df):
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Extract time-based features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month

    # Add new features
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['season'] = df['month'].map({
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
    })

    # Aggregate temperature and humidity columns
    df['avg_temp'] = df[['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']].mean(axis=1)
    df['avg_humidity'] = df[['rh_1', 'rh_2', 'rh_3', 'rh_4', 'rh_5', 'rh_6', 'rh_7', 'rh_8', 'rh_9']].mean(axis=1)

    # Drop unnecessary columns
    df = df.drop(columns=[
        'date', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9',
        'rh_1', 'rh_2', 'rh_3', 'rh_4', 'rh_5', 'rh_6', 'rh_7', 'rh_8', 'rh_9',
        'rv1', 'rv2'
    ])

    # Log-transform the target variable
    df['appliances_log'] = np.log(df['appliances'] + 1)

    return df

# Evaluate the model
def evaluate_model(model, preprocessor, X_test, y_test):
    # Preprocess the test data
    X_test_transformed = preprocessor.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_transformed)

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")

# Main function
def main():
    # Load data
    filepath = 'data.csv'
    df = load_data(filepath)

    # Preprocess data
    df = preprocess_data(df)

    # Define features and target
    X = df.drop(columns=['appliances', 'appliances_log'])
    y = df['appliances_log']

    # Load the trained model and preprocessor
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')

    # Evaluate the model
    evaluate_model(model, preprocessor, X, y)

if __name__ == '__main__':
    main()