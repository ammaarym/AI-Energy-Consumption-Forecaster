import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Load the energy consumption data (replace with your dataset)
data = pd.read_csv('energy_consumption_data.csv')

# Assuming 'timestamp' is the timestamp column and 'consumption' is the energy consumption data
# You may have additional features or data to consider in a real-world scenario

# Data preprocessing
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.rename(columns={'timestamp': 'ds', 'consumption': 'y'})

# Handling missing data (replace NaNs with interpolated values)
data = data.interpolate(method='linear')

# Feature engineering (e.g., add holidays and other external factors)
# Example: Define holidays
holidays = pd.DataFrame({
    'holiday': 'custom_holiday',
    'ds': pd.to_datetime(['2022-01-01', '2022-07-04']),  # Add your holiday dates
    'lower_window': 0,
    'upper_window': 1,
})

# Initialize and configure the Prophet model
model = Prophet(
    seasonality_mode='additive',  # Modify as needed (e.g., 'multiplicative')
    holidays=holidays,  # Add your defined holidays here
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
)

# Fit the Prophet model to the data
model.fit(data)

# Create a dataframe for future prediction
future = model.make_future_dataframe(periods=365)  # Adjust the prediction horizon as needed

# Make predictions
forecast = model.predict(future)

# Visualize the forecast
fig = model.plot(forecast)
plt.title("Energy Consumption Forecast")
plt.xlabel("Time")
plt.ylabel("Consumption")
plt.show()

# Access forecasted values
forecasted_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Evaluate the model (e.g., using metrics like MAE, MSE, etc.)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(data['y'], forecasted_data['yhat'])
print(f"Mean Absolute Error (MAE): {mae}")

# You can further enhance this script by optimizing hyperparameters, including more external factors, and performing cross-validation.
