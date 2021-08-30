from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from statsmodels.tsa.vector_ar.var_model import VAR


data = pd.read_csv("Earthquake_db.csv", delimiter=',', index_col='Date', parse_dates=True)


def testing_stationary(timeseries):
    # Determing rolling statistics
    roll_mean = timeseries.rolling(12).mean()
    roll_std = timeseries.rolling(12).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(roll_mean, color='red', label='Rolling Mean')
    std = plt.plot(roll_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    #plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')

    # The number of lags is chosen to minimise the Akaike's Information Criterion (AIC)
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

    print('\n\n')

    # Perform KPSS test:
    print('Results of KPSS Test:')

    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])

    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)


testing_stationary(data["Depth"])
# testing_stationary(data["Magnitude"])
# testing_stationary(data["Latitude"])
# testing_stationary(data["Longitude"])

# Make data stationary
# 1st difference
df_differenced = data.diff().dropna()
testing_stationary(df_differenced["Depth"])
testing_stationary(df_differenced["Magnitude"])
testing_stationary(df_differenced["Latitude"])
testing_stationary(df_differenced["Longitude"])

# Train-test part
train, test = df_differenced.iloc[:23158, :], data.iloc[23158:, :]

# Fit the model
model = VAR(endog=train.astype(float), freq=None)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(test))

cols = data.columns
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,4):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

for i in cols:
    print('RMSE value for', i, 'is : ', sqrt(mean_squared_error(pred[i], test[i])))

p_mag = pd.DataFrame(test['Magnitude'])
p_mag['pred'] = pred['Magnitude'].values
p_mag.plot(legend=True)
plt.show()

rmse_var_pred = sqrt(mean_squared_error(p_mag['Magnitude'], p_mag['pred']))
print("RMSE", rmse_var_pred)
var_score = r2_score(p_mag['Magnitude'], p_mag['pred'])
print("r2 score ", var_score)
var_ms = mean_squared_error(p_mag['Magnitude'], p_mag['pred'])
print("mean squared error ", var_ms)
var_m = mean_absolute_error(p_mag['Magnitude'], p_mag['pred'])
print("mean absolute error ", var_m)


