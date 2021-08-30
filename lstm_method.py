# remove tenserflow info from console
import os
from math import sqrt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Bidirectional
from keras.layers import Dropout
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("Earthquake_db.csv",  delimiter=',', parse_dates=['Date'], index_col="Date")

train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))

f_columns = ['Latitude', 'Longitude', 'Depth']

f_transformer = RobustScaler()
cnt_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
cnt_transformer = cnt_transformer.fit(train[['Magnitude']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['Magnitude'] = cnt_transformer.transform(train[['Magnitude']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['Magnitude'] = cnt_transformer.transform(test[['Magnitude']])


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10

# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train, train['Magnitude'], time_steps)
X_test, y_test = create_dataset(test, test['Magnitude'], time_steps)
print(X_train.shape, y_train.shape)

model = Sequential()
model.add(Bidirectional(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train,epochs=30, batch_size=32, validation_split=0.1, shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))

# convert y_test and y_pred to dataframe
y_df = pd.DataFrame(y_test_inv.T, columns=['y_test_mag'])
y_df['y_pred_mag'] = y_pred_inv

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Magnitude')
plt.xlabel('Time Step')
plt.legend()
plt.show()

plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Magnitude')
plt.xlabel('Time Step')
plt.legend()
plt.show()

# Evaluation
rmse_lstm_pred = sqrt(mean_squared_error(y_df['y_test_mag'], y_df['y_pred_mag']))
print("LSTM RMSE", rmse_lstm_pred)
lstm_score = r2_score(y_df['y_test_mag'], y_df['y_pred_mag'])
print("LSTM r2 score ", lstm_score)
lstm_ms = mean_squared_error(y_df['y_test_mag'], y_df['y_pred_mag'])
print("LSTM mean squared error ", lstm_ms)
lstm_m = mean_absolute_error(y_df['y_test_mag'], y_df['y_pred_mag'])
print("LSTM mean absolute error ", lstm_m)



# https://towardsdatascience.com/demand-prediction-with-lstms-using-tensorflow-2-and-keras-in-python-1d1076fc89a0


# train_df,test_df = data_df[0:-len(df_targets)], data_df[-len(df_targets):]
#
# x = data_df.iloc[:,:-1].values # inputs
# y = data_df.iloc[:,-1].values  # target
#
# sc = StandardScaler()
# x_scale = sc.fit_transform(x)
# y_scale = sc.fit_transform(y.reshape(-1,1))
#
# df_targets = data_df["2020-03-31 00:00:00 ":"2020-04-01 00:00:00"]
# x_train = data_df.iloc[0:-len(df_targets), :-1]
# y_train = data_df.iloc[:-len(df_targets), -1]
# x_test = data_df.iloc[-len(df_targets):, :-1]
# y_test = data_df.iloc[-len(df_targets):, -1]
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)




