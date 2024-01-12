import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def calculate_rmse(forecast, cycling_test):
    # Filter forecast and cycling_test based on date
    predictions = forecast[forecast['ds'] >= '2018']
    predictions['ds'] = pd.to_datetime(predictions['ds'])

    truth = cycling_test
    truth['ds'] = pd.to_datetime(truth['ds'])

    # Merge predicted and true values on the 'ds' column
    merged_df = pd.merge(predictions, truth, on='ds', how='inner', suffixes=('_pred', '_true'))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(merged_df['y_true'], merged_df['yhat']))

    print(f"Root Mean Squared Error (RMSE): {rmse}")