import os
from buynomics.src.utils.functions import preprocess_data, train_data
from buynomics.src.utils.train_model import train_model

if __name__ == '__main__':
    
    current_file = os.path.dirname(__file__)
    sales_path = os.path.join(current_file,"..","..","data","sales.csv")
    product_path = os.path.join(current_file,"..","..","data","product.csv")
    models_path = os.path.join(current_file,"..","models")

    df = preprocess_data(sales_path, product_path)

    X, y = train_data(df)

    use_model = ['Linear Regression', 'LightGBM', 'XGBoost']

    for model_name in use_model:
        print(f"Training {model_name}\n")
        train_model(X, y, model_name, models_path)
        print("\n")

