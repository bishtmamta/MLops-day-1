import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# load the dataset
df = pd.read_csv("C:\\Users\\princ\\MLops day 1\\data\\Advertising.csv")

# train test split
x,y=df.drop(columns=['sales']),df['sales']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# model dump
joblib.dump(model, "models\\linear_regression_model.pkl")
