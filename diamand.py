import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
diamond = pd.read_csv('Diamonds Prices2022.csv')


print(diamond.head())
print(diamond.info())
print(diamond.describe())



plt.scatter(diamond['carat'], diamond['price'], color='blue',  label='Actual Data')
plt.xlabel("Carat")
plt.ylabel("Price")
plt.title('Diamond Prices Carat')


X = diamond[['carat']]
Y = diamond['price']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


model_lr = LinearRegression()
model_lr.fit(X_train, Y_train)

a_lr = model_lr.coef_[0]
b_lr = model_lr.intercept_


Y_pred_lr = model_lr.predict(X_test)


plt.plot(X_test, Y_pred_lr, color='red', linewidth=2, label=f'Linear Regression: f(x) = {a_lr:.2f}x + {b_lr:.2f}')
plt.show()
# Decision Tree Regressor
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, Y_train)


Y_pred_dt = model_dt.predict(X_test)




model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, Y_train)


Y_pred_rf = model_rf.predict(X_test)




mse_lr = mean_squared_error(Y_test, Y_pred_lr)
r2_lr = r2_score(Y_test, Y_pred_lr)
print("\nLinear Regression:")
print("Mean Squared Error:", mse_lr)
print("R² Score:", r2_lr)


mse_dt = mean_squared_error(Y_test, Y_pred_dt)
r2_dt = r2_score(Y_test, Y_pred_dt)
print("\nDecision Tree Regressor:")
print("Mean Squared Error:", mse_dt)
print("R² Score:", r2_dt)


mse_rf = mean_squared_error(Y_test, Y_pred_rf)
r2_rf = r2_score(Y_test, Y_pred_rf)
print("\nRandom Forest Regressor:")
print("Mean Squared Error:", mse_rf)
print("R² Score:", r2_rf)
