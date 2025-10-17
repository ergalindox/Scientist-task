
from sklearn.model_selection import  cross_validate
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import sklearn


def train_model(X,y, model_name, save_path):
  """
  Function trains a model and prints the scores with the testing data.
  Args: 
  - X: dataframe with all columns except the one to be predicted
  - y: dataframe with only the column to be predicted
  - model_name: the model type that will be used
  - save_path: where the model will be saved for future predictions
  """

  if model_name == "Linear Regression":
    model = LinearRegression()
    file_model = 'Linear_Reg.pkl'
  elif model_name == "LightGBM":
    model = lgb.LGBMRegressor(verbosity = -1)
    file_model = 'LGBMReg.pkl'
  elif model_name == "XGBoost":
    model = xgb.XGBRegressor()
    file_model = 'XGBoost_Reg.pkl'

  
  scores = cross_validate(model, X, y, scoring = ["r2", "neg_mean_absolute_error"], cv = 8)
  model.fit(X, y)

  
  print(f"CV R2 Mean: {scores['test_r2'].mean():.4f}")
  print(f"CV MAE Mean: {(scores['test_neg_mean_absolute_error'].mean())*-1:.4f}")

  filename = file_model
  save_path = os.path.join(save_path, filename)
  with open(save_path, 'wb') as file:
      pickle.dump(model, file)

  

  

