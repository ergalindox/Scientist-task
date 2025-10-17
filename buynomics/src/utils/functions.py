import pandas as pd

def preprocess_data(sales_path, product_path):
  """
  Function that preprocess dataframes. 
  Args: 
   - sales_path (string): path to the sales dataset
   - product_path (string): path to the products dataset
  """
  products = pd.read_csv(product_path)
  sales = pd.read_csv(sales_path)
  products=products.drop("Unnamed: 0", axis=1)
  sales=sales.drop("Unnamed: 0", axis=1)
  final_df = sales.merge(products, on='product_id', how='left')
  final_df['Avg_Price_by_Brand'] = final_df.groupby('brand')['price'].transform('mean')
  final_df['Avg_Price_by_Product'] = final_df.groupby('product_id')['price'].transform('mean')
  final_df['Total_volum_pack'] = final_df['volume_per_joghurt_g'] * final_df['packsize']
  final_df['date'] = pd.to_datetime(final_df['date'])
  final_df.drop("volume_per_joghurt_g", axis=1)

  return final_df


def train_data(final_df):
  """
  Function returns the final two dataframes to be trained in the model 
  Args: 
   - final_df: dataframe already processed with all necessary columns
  """

  num_cols =  final_df.select_dtypes(include="number").columns.tolist()
  cat_cols = final_df.select_dtypes(include="object").columns.tolist()

  # Move 'packsize' from numeric to categorical
  if 'packsize' in num_cols:
      num_cols.remove('packsize')
      cat_cols.append('packsize')

  df_encoded = pd.get_dummies(final_df, columns=cat_cols, drop_first=True)
  feature_cols = [col for col in df_encoded.columns if any(cat in col for cat in cat_cols)] + [col for col in num_cols if col != 'units']
  X = df_encoded[feature_cols]
  y = df_encoded['units']

  return X,y