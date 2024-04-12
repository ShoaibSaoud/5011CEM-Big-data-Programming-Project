#D

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

# Function to load and preprocess data
def load_and_preprocess_data(file_path, date_col, week_col, target_week):
    data_frame = pd.read_csv(file_path)
    data_frame[date_col] = pd.to_datetime(data_frame[date_col], errors='coerce')
    filtered_data = data_frame[data_frame[week_col] == target_week]
    return filtered_data

# Function to fit and evaluate models
def fit_and_evaluate_models(features_train, features_test, targets_train, targets_test):
    regression_models = {}

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(features_train, targets_train)
    regression_models['linear'] = {
        'model': linear_model,
        'predictions': {
            'train': linear_model.predict(features_train),
            'test': linear_model.predict(features_test)
        },
        'score_train': linear_model.score(features_train, targets_train),
        'score_test': linear_model.score(features_test, targets_test)
    }

    # Polynomial Regression
    polynomial_transformer = PolynomialFeatures(degree=2, include_bias=False)
    features_train_poly = polynomial_transformer.fit_transform(features_train)
    features_test_poly = polynomial_transformer.transform(features_test)
    polynomial_model = LinearRegression()
    polynomial_model.fit(features_train_poly, targets_train)
    regression_models['polynomial'] = {
        'model': polynomial_model,
        'transformer': polynomial_transformer,
        'predictions': {
            'train': polynomial_model.predict(features_train_poly),
            'test': polynomial_model.predict(features_test_poly)
        },
        'score_train': polynomial_model.score(features_train_poly, targets_train),
        'score_test': polynomial_model.score(features_test_poly, targets_test)
    }

    # OLS model
    features_train_ols = sm.add_constant(features_train)
    ols_model = sm.OLS(targets_train, features_train_ols).fit()
    features_test_ols = sm.add_constant(features_test)
    regression_models['OLS'] = {
        'model': ols_model,
        'predictions': {
            'train': ols_model.predict(features_train_ols),
            'test': ols_model.predict(features_test_ols)
        },
        'summary': ols_model.summary()
    }

    return regression_models

# Function to create summary table
def create_summary_table(models):
    summary_data = {
        "Model": ["Linear Regression", "Polynomial Regression"],
        "R-squared (Training)": [models['linear']['score_train'], models['polynomial']['score_train']]
        
    }
    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Function to plot results
def plot_regression_results(features_train, targets_train, model, transformer, model_label, plot_title):
    plt.figure(figsize=(10, 6))
    plt.scatter(features_train, targets_train, color='darkred', label='Actual data')
    plt.xlabel('Trips 1-3 Miles', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Trips >=500', fontsize=12, fontweight='bold')
    plt.title(plot_title, fontsize=14, fontweight='bold')

    if model_label == 'Polynomial':
        feature_range = np.linspace(features_train.min(), features_train.max(), 300).reshape(-1, 1)
        feature_range_poly = transformer.transform(feature_range)
        plt.plot(feature_range, model.predict(feature_range_poly), color='limegreen', linewidth=2, label='Polynomial Fit')
    else:
        plt.plot(features_train, model.predict(features_train), color='limegreen', linewidth=2, label='Linear Fit')
    
    plt.legend()
    plt.grid(True)
    plt.show()

# Main script
if __name__ == "__main__":
    full_trip_data = load_and_preprocess_data("Trips_Full_Data.csv", 'Date', 'Week of Date', 32)
    distance_trip_data = load_and_preprocess_data("Trips_by_Distance.csv", 'Date', 'Week', 32)

    if not full_trip_data.empty and not distance_trip_data.empty and 'Trips 1-3 Miles' in full_trip_data.columns and 'Number of Trips >=500' in distance_trip_data.columns:
        min_length = min(len(full_trip_data), len(distance_trip_data))
        trip_features = full_trip_data['Trips 1-3 Miles'].iloc[:min_length].values.reshape(-1, 1)
        trip_targets = distance_trip_data['Number of Trips >=500'].iloc[:min_length].values
        features_train, features_test, targets_train, targets_test = train_test_split(trip_features, trip_targets, test_size=0.2, random_state=42)
        
        models = fit_and_evaluate_models(features_train, features_test, targets_train, targets_test)
        models_summary = create_summary_table(models)
        
        print("Model Summary Table:")
        print(models_summary)
        
        print("\nPredictions from Polynomial Regression (Testing):")
        print(models['polynomial']['predictions']['test'])
        
        print("\nOLS Regression Results:")
        print(models['OLS']['summary'])

        plot_regression_results(features_train, targets_train, models['linear']['model'], None, 'Linear', 'Linear Regression')
        plot_regression_results(features_train, targets_train, models['polynomial']['model'], models['polynomial']['transformer'], 'Polynomial', 'Polynomial Regression')
    else:
        print("Check the input data and column names.")
