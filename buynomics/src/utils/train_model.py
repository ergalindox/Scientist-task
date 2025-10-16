
from sklearn.model_selection import  cross_validate
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import sklearn


def train_model(X,y, model_name, save_path):
  
  if model_name == "Linear Regression":
    model = LinearRegression()
    file_model = 'Linear_Reg.pkl'
  elif model_name == "LightGBM":
    model = lgb.LGBMRegressor(verbosity = -1)
    file_model = 'LGBMReg.pkl'
  elif model_name == "XGBoost":
    model = xgb.XGBRegressor(n_estimators=100,          # number of boosting rounds
                            learning_rate=0.03,         # smaller learning rate for better generalization
                            max_depth=6,                # tree depth (controls complexity)
                            min_child_weight=5,         # prevents overfitting by requiring min sum of instance weight
                            subsample=0.8,              # row sampling for each tree
                            colsample_bytree=0.8,       # column sampling for each tree
                            gamma=0.1,                  # minimum loss reduction required for a split
                            reg_alpha=0.1,              # L1 regularization (sparsity)
                            reg_lambda=1.0,             # L2 regularization
                            objective='reg:squarederror',
                            booster='gbtree')
    file_model = 'XGBoost_Reg.pkl'

  
  scores = cross_validate(model, X, y, scoring = ["r2", "neg_mean_absolute_error"], cv = 8)
  model.fit(X, y)

  
  print(f"CV R2 Mean: {scores['test_r2'].mean():.4f}")
  print(f"CV MAE Mean: {(scores['test_neg_mean_absolute_error'].mean())*-1:.4f}")

  filename = file_model
  save_path = os.path.join(save_path, filename)
  with open(save_path, 'wb') as file:
      pickle.dump(model, file)

  

  

