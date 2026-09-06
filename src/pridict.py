import mlflow
import mlflow.sklearn
import pandas as  pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# load saved model
model= mlflow.sklearn.load_model(
    "models:/Sales_Prediction_Model@champion"
)

# new obeservation 
new_data= pd.DataFrame({"TV":[35],"radio":[50000],"newspaper":[8]})

# pridiction
pridiction= model.predict(new_data)
print("Predicted sales:", pridiction[0])   

