
from sklearn.model_selection import  cross_validate
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import pickle
import os


def train_model(X,y, model_name, save_path):
  
  if model_name == "Linear Regression":
    model = LinearRegression()
    file_model = 'Linear_Reg.pkl'
  elif model_name == "LightGBM":
    model = lgb.LGBMRegressor()
    file_model = 'LGBMReg.pkl'
  elif model_name == "XGBoost":
    model = xgb.XGBRegressor()
    file_model = 'XGBoost_Reg.pkl'

  scores = cross_validate(model, X, y, scoring = ["r2", "neg_mean_absolute_error"], cv = 8)
  model.fit(X, y)

  
  print(f"CV R2 Mean: {scores['test_r2'].mean():.4f}")
  print(f"CV MAE Mean: {scores['test_neg_mean_absolute_error'].mean():.4f}")

  filename = file_model
  save_path = os.path.join(save_path, filename)
  with open(save_path, 'wb') as file:
      pickle.dump(model, file)

  

  

