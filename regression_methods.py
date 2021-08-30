import glob
import datetime
import time
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def merge_dataset():
    # Merge all txt files into data frame
    txt_files = glob.glob("C:/PycharmProjects/CMP5103-Project/*.txt")
    temp_list = []
    for t in txt_files:
        data = pd.read_csv(t, sep='\t', index_col=False)
        temp_list.append(data)
    data = pd.concat(temp_list, axis=0, ignore_index=True)
    df = pd.DataFrame(data)

    # drop column Yer
    data = data.drop(['Yer'], axis=1)

    df.to_csv("Earthquake_db.csv", index=False, header=True)
    print("done")

# IN CSV column names translate to english
data = pd.read_csv("Earthquake_db.csv",  delimiter=',')
print(data.shape)
# convert object to timestamp numeric
timestamp = []
for d in data['Date']:
    ts = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
    timestamp.append(time.mktime(ts.timetuple()))

timeStamp = pd.Series(timestamp)
data['Timestamp'] = timeStamp.values

data = data[['Timestamp', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

X = data.iloc[:, -1:].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
x_train_scale = sc.fit_transform(X_train)
sc2 = StandardScaler()
y_train_scale = np.ravel(sc2.fit_transform(y_train.reshape(-1,1)))


# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
l_pred = lr.predict(X_test)
l_df = pd.DataFrame({'Actual': y_test, 'Predicted': l_pred})
rmse_l_pred = sqrt(mean_squared_error(y_test, l_pred))
print("Linear Regression RMSE", rmse_l_pred)
lr_score = r2_score(y_test, l_pred)
print("Linear Regression r2 score ", lr_score)
dt_ms = mean_squared_error(y_test, l_pred)
print("Linear Regression  mean squared error ", dt_ms)
lr_m = mean_absolute_error(y_test, l_pred)
print("Linear Regression  mean absolute error ", lr_m)

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X_train)
lr2 = LinearRegression()
lr2.fit(x_poly, y_train)
poly_reg = lr2.predict(poly_reg.fit_transform(X_test))
p_df = pd.DataFrame({'Actual': y_test, 'Predicted': poly_reg})
rmse_p_reg = sqrt(mean_squared_error(y_test, poly_reg))
print("Polynomial Regression RMSE", rmse_p_reg)
pr_score = r2_score(y_test, poly_reg)
print("Polynomial Regression r2 score ", pr_score)
pr_ms = mean_squared_error(y_test, poly_reg)
print("Polynomial Regression  mean squared error ", pr_ms)
pr_m = mean_absolute_error(y_test, poly_reg)
print("Polynomial Regression  mean absolute error ", pr_m)


# Decision Tree
r_dt = DecisionTreeRegressor(random_state=0, max_depth=2)
r_dt.fit(X_train, y_train)
d_pred = r_dt.predict(X_test)
d_df = pd.DataFrame({'Actual': y_test, 'Predicted': d_pred})
rmse_d_pred = sqrt(mean_squared_error(y_test, d_pred))
print("Decision Tree RMSE", rmse_d_pred)
dt_score = r2_score(y_test, d_pred)
print("Decision Tree r2 score ", dt_score)
dt_ms = mean_squared_error(y_test, d_pred)
print("Decision Tree  mean squared error ", dt_ms)
dt_m = mean_absolute_error(y_test, d_pred)
print("Decision Tree  mean absolute error ", dt_m)

# Random Forest
# max depth versus error
md = 20
md_errors = np.zeros(md)
# random forest regression
for i in range(1, md + 1):
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=i, random_state=0)
    rf_reg.fit(X_train, y_train)
    r_pred = rf_reg.predict(X_test)
    # finding error
    md_errors[i - 1] = sqrt(mean_squared_error(y_test, r_pred))
r_df = pd.DataFrame({'Actual': y_test, 'Predicted': r_pred})
print("Random Forest RMSE ", md_errors[i - 1])
rf_score = r2_score(y_test, r_pred)
print("Random Forest r2 score ", rf_score)
rf_ms = mean_squared_error(y_test, r_pred)
print("Random Forest mean squared error ", rf_ms)
rf_m = mean_absolute_error(y_test, r_pred)
print("Random Forest  mean absolute error ", rf_m)

# SVR
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(x_train_scale, y_train_scale)
svr_pred = svr.predict(X_test)
svr_df = pd.DataFrame({'Actual': y_test, 'Predicted': svr_pred})
rmse_svr_pred = sqrt(mean_squared_error(y_test, svr_pred))
print("SVR RMSE", rmse_svr_pred)
svr_score = r2_score(y_test, svr_pred)
print("SVR r2 score ", svr_score)
svr_ms = mean_squared_error(y_test, svr_pred)
print("SVR mean squared error ", svr_ms)
svr_m = mean_absolute_error(y_test, svr_pred)
print("SVR mean absolute error ", svr_m)


# Visualize Results
# Linear Regression
plt.figure(figsize=(6, 5))
plt.scatter(l_df['Actual'], l_df['Predicted'], color='b')
plt.plot([min(l_df['Actual']), max(l_df['Actual'])], [min(l_df['Predicted']), max(l_df['Actual'])], '--k')
plt.axis('tight')
plt.title("Linear Regression true and predicted value comparison")
plt.xlabel("y_test")
plt.ylabel("pred")
plt.tight_layout()
plt.show()

# Polynomial
plt.figure(figsize=(6, 5))
plt.scatter(p_df['Actual'], p_df['Predicted'], color='r')
plt.plot([min(p_df['Actual']), max(p_df['Actual'])], [min(p_df['Predicted']), max(p_df['Actual'])], '--k')
plt.axis('tight')
plt.title("Polynomial Regression true and predicted value comparison")
plt.xlabel("y_test")
plt.ylabel("pred")
plt.tight_layout()
plt.show()

# Decision Tree
plt.figure(figsize=(6, 5))
plt.scatter(d_df['Actual'], d_df['Predicted'], color='g')
plt.plot([min(d_df['Actual']), max(d_df['Actual'])], [min(d_df['Predicted']), max(d_df['Actual'])], '--k')
plt.axis('tight')
plt.title("Decision Tree true and predicted value comparison")
plt.xlabel("y_test")
plt.ylabel("pred")
plt.tight_layout()
plt.show()

# Random
plt.figure(figsize=(6, 5))
plt.scatter(r_df['Actual'], r_df['Predicted'], color='y')
plt.plot([min(r_df['Actual']), max(r_df['Actual'])], [min(r_df['Predicted']), max(r_df['Actual'])], '--k')
plt.axis('tight')
plt.title("Random Forest true and predicted value comparison")
plt.xlabel("y_test")
plt.ylabel("pred")
plt.tight_layout()
plt.show()

# SVR
plt.figure(figsize=(6, 5))
plt.scatter(svr_df['Actual'], svr_df['Predicted'], color='b')
plt.plot([min(svr_df['Actual']), max(svr_df['Actual'])], [min(svr_df['Predicted']), max(svr_df['Actual'])], '--k')
plt.axis('tight')
plt.title("SVR true and predicted value comparison")
plt.xlabel("y_test")
plt.ylabel("pred")
plt.tight_layout()
plt.show()