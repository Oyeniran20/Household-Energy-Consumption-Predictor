# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Columns in the dataset:", df.columns.tolist())  # Debugging step
    return df


# Preprocess the data
def preprocess_data(df):
    print("\nStarting data preprocessing...")
    
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
    
    
    print("Preprocessing completed successfully! New shape:", df.shape)
    return df

# Train the model
def train_model(X_train, y_train):
    print("\nTraining model with RandomForestRegressor...")

    
    # Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    print("Model Parameters:", model.get_params())

    return model

# Save the model and preprocessor
def save_artifacts(model, preprocessor, model_path, preprocessor_path):
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")

# Main function
def main():
    # Load data
    filepath = 'C:/Users/User/Documents/Data Science/My_Learning/Energy Project/data.csv'
    df = load_data(filepath)

    # Preprocess data
    df = preprocess_data(df)

    # Define features and target
    X = df.drop(columns=['appliances', 'appliances_log'])
    y = df['appliances_log']

    # Define numeric and categorical columns
    num_cols = X.select_dtypes(include=np.number).columns.to_list()
    cat_cols = X.select_dtypes(include='object').columns.to_list()
    
    print(f"\nFeatures selected: {len(num_cols)} numeric, {len(cat_cols)} categorical.")

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")


    # Preprocess the training data
    X_train = preprocessor.fit_transform(X_train)

    # Train the model
    model = train_model(X_train, y_train)

    # Save the model and preprocessor
    save_artifacts(model, preprocessor, 'model.pkl', 'preprocessor.pkl')

if __name__ == '__main__':
    main()
