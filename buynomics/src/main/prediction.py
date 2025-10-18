import os
import pickle
import copy
from buynomics.src.utils.functions import preprocess_data, train_data
from buynomics.src.utils.model_predict import prediction

if __name__ == '__main__':
    
    model_name ='LGBMReg'
    current_file = os.path.dirname(__file__)

    models_path = os.path.join(current_file,"..","models")
    
    sales_path = os.path.join(current_file,"..","..","data","sales.csv")
    product_path = os.path.join(current_file,"..","..","data","product.csv")
    df = preprocess_data(sales_path, product_path)

    if model_name == 'XGBoost':
        model_path = os.path.join(current_file,"..","models","XGBoost_Reg.pkl")

    elif model_name == 'LGBMReg':
        model_path = os.path.join(current_file,"..","models","LGBMReg.pkl")

    elif model_name == 'Linear_Reg':
        model_path = os.path.join(current_file,"..","models","Linear_Reg.pkl")

    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please ensure the model is saved first.")

    
    
    save_path = os.path.join(current_file,"..","..","output","output.csv")
    prediction(model, df, save_path)
    



    
