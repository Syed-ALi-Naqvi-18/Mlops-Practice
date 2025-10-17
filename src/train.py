import pandas as pd
import yaml
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_params(params_file):
    """Load parameters from YAML file"""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_data(data_path):
    """Load the Pakistan House Price dataset"""
    # For now, we'll create a mock dataset since we don't have the actual dataset
    # In a real scenario, you would load the actual dataset from data_path
    print(f"Loading data from {data_path}")
    
    # Creating a mock dataset for demonstration
    data = {
        'price': [1000000, 2000000, 1500000, 3000000, 2500000, 1800000, 2200000, 2800000],
        'area': [1200, 2000, 1500, 3000, 2500, 1800, 2200, 2800],
        'bedrooms': [3, 4, 3, 5, 4, 3, 4, 5],
        'baths': [2, 3, 2, 4, 3, 2, 3, 4],
        'city': ['Karachi', 'Lahore', 'Islamabad', 'Karachi', 'Lahore', 'Islamabad', 'Karachi', 'Lahore'],
        'province_name': ['Sindh', 'Punjab', 'Punjab', 'Sindh', 'Punjab', 'Punjab', 'Sindh', 'Punjab']
    }
    
    return pd.DataFrame(data)

def preprocess_data(df, params):
    """Preprocess the data"""
    print("Preprocessing data...")
    
    # Encode categorical variables
    label_encoders = {}
    for col in params['data']['categorical_columns']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    return df, label_encoders

def train_model(X_train, y_train, params):
    """Train the model"""
    print("Training model...")
    
    if params['model_type'] == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=params['random_forest']['n_estimators'],
            max_depth=params['random_forest']['max_depth'],
            min_samples_split=params['random_forest']['min_samples_split'],
            min_samples_leaf=params['random_forest']['min_samples_leaf'],
            random_state=params['random_seed']
        )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    
    return {'mse': mse, 'r2': r2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='params.yaml', help='Path to params.yaml')
    parser.add_argument('--data', type=str, default='data/Pakistan_House_Price.csv', help='Path to dataset')
    parser.add_argument('--model-out', type=str, default='model/model.pkl', help='Path to save model')
    parser.add_argument('--metrics-out', type=str, default='metrics.json', help='Path to save metrics')
    
    args = parser.parse_args()
    
    # Load parameters
    params = load_params(args.params)
    
    # Load data
    df = load_data(args.data)
    
    # Preprocess data
    df, label_encoders = preprocess_data(df, params)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != params['data']['target_column']]
    X = df[feature_cols]
    y = df[params['data']['target_column']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['test_size'], 
        random_state=params['random_seed']
    )
    
    # Train model
    model = train_model(X_train, y_train, params)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    joblib.dump(model, args.model_out)
    print(f"Model saved to {args.model_out}")
    
    # Save metrics
    import json
    with open(args.metrics_out, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {args.metrics_out}")

if __name__ == "__main__":
    main()