import joblib
import pandas as pd
import numpy as np

model = joblib.load("models\\linear_regression_model.pkl")
new_data = pd.DataFrame([[123,56,89]])
predictions = model.predict(new_data)
print("Predicted sales:", predictions)