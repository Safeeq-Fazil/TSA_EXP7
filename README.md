### Developed by: Safeeq Fazil A
### Reg no: 212222240086
### Date: 28.09.2024
# Ex.No: 07                                       AUTO REGRESSIVE MODEL




### AIM:
To Implemented an Auto Regressive Model using Python for the vegetable dataset.

### ALGORITHM:

1.Load the dataset, convert 'Date' to datetime, filter for the commodity, resample to monthly averages, and handle missing values.

2.Plot ACF and PACF to identify the appropriate lag order for the AR model.

3.Fit the AR model using the training data with the selected lag order based on PACF.

4.Predict the last 12 months, calculate MSE, and plot the actual test data against the predicted values.

5.Forecast the next 12 months beyond the test period and visualize the final predictions along with the training data.

### PROGRAM
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = '/content/vegetable.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Select data for a specific commodity, e.g., 'Tomato Big(Nepali)'
commodity_data = data[data['Commodity'] == 'Tomato Big(Nepali)']

# Set 'Date' as the index
commodity_data.set_index('Date', inplace=True)

# Resample the data to monthly averages
monthly_data = commodity_data['Average'].resample('M').mean()

# Drop any NaN values
monthly_data = monthly_data.dropna()

# Split data into training and test sets (we use the last 12 months as test data)
train_data = monthly_data[:-12]
test_data = monthly_data[-12:]

# Plot ACF and PACF to find the AR order
plt.figure(figsize=(10, 6))
plt.subplot(211)
plot_acf(train_data, lags=20, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')

plt.subplot(212)
plot_pacf(train_data, lags=20, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# From the PACF plot, we can determine the AR order (choose p from PACF plot)
# Let's assume we pick an AR order of 2 based on the PACF

# Fit the AR model (Auto Regressive model)
ar_order = 2
ar_model = AutoReg(train_data, lags=ar_order).fit()

# Test Predictions for the last 12 months
test_predictions = ar_model.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

# Calculate the Mean Squared Error (MSE) between the test data and predictions
mse = mean_squared_error(test_data, test_predictions)
print(f'Test Mean Squared Error: {mse}')

# Final Forecast for the next 12 months beyond the test period
final_forecast = ar_model.predict(start=len(train_data)+len(test_data), end=len(train_data)+len(test_data)+11)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot the training data
plt.plot(train_data, label='Training Data', color='blue')

# Plot the actual test data
plt.plot(test_data, label='Test Data', color='orange')

# Plot the test predictions
plt.plot(test_predictions, label='Test Predictions', color='green')

# Plot the final forecast
plt.plot(final_forecast, label='Final Forecast (Next 12 months)', color='red')

plt.title('Auto Regressive Model for Tomato Big (Nepali)')
plt.legend(loc='best')
plt.show()


```
### OUTPUT:

GIVEN DATA:

## PACF - ACF
![image](https://github.com/user-attachments/assets/c7b7b325-60a3-4c4f-9a2b-a73a4424e1ed)



## PREDICTION & FINIAL PREDICTION
![image](https://github.com/user-attachments/assets/5cc3eace-6ac5-4e74-b254-cdcf77b2b827)
### Mean Squared Error:
![image](https://github.com/user-attachments/assets/21137651-acc6-41aa-80e0-301355caf4c2)


### RESULT:
Thus the python program for implemented the auto regression function for the vegetable dataset is executed succesfully.
