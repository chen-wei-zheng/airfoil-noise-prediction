import pandas as pd
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Tuple

DATA_URL = "https://archive.ics.uci.edu/static/public/291/data.csv"

def download_and_load_data(url: str) -> pd.DataFrame:
    """Downloads, loads, and preprocesses the airfoil self-noise data."""
    print(f"--- Downloading data from {url} ---")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Load data from the response text
        data = pd.read_csv(io.StringIO(response.text), sep=",", header=0)

        # Rename columns to be more Python-friendly
        data.columns = [
            "frequency",
            "angle_of_attack",
            "chord_length",
            "free_stream_velocity",
            "suction_side_thickness",
            "sound_pressure_level"
        ]
        print("--- Data Loaded and Cleaned Successfully ---")
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Initializes and trains the Random Forest Regressor model."""
    print("\n--- Training Random Forest Model ---")
    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    model.fit(X_train, y_train)
    print("Model training complete.")
    print(f"Model OOB Score: {model.oob_score_:.4f}")
    return model

def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluates the model and prints performance metrics."""
    print("\n--- Evaluating Model Performance ---")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

def show_feature_importance(model: RandomForestRegressor, features: pd.Index):
    """Displays the feature importances of the trained model."""
    print("\n--- Feature Importances ---")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    print(feature_importance_df)

def main():
    """Main function to orchestrate the project pipeline."""
    # Load the data
    airfoil_data = download_and_load_data(DATA_URL)
    if airfoil_data.empty:
        return # Exit if data loading failed

    # Separate features (X) and target (y)
    X = airfoil_data.drop('sound_pressure_level', axis=1)
    y = airfoil_data['sound_pressure_level']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n--- Data Split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples ---")

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Show feature importances
    show_feature_importance(model, X.columns)

if __name__ == "__main__":
    main()