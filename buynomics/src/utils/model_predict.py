
import copy
from buynomics.src.utils.functions import train_data


def prediction(model, df, save_path):
    """
    Function makes a prediction with a trained model to give simulated units depending on an input price change
    Args: 
    - model: model already trained with data
    - df: dataframe with processed data that was used to train the model
    - save_path: path to save the final dataset with predictions
    """

    X, _ = train_data(df)
    simulated_df = copy.deepcopy(X)
    feature_cols = list(X.columns)

    # Ask user for price adjustment
    factor_value = float(input("Enter desired price level as a percentage (e.g., 120 = +20%, 80 = -20%): "))

    # Apply new price
    simulated_df['price'] = simulated_df['price'] * (factor_value / 100)

    # Predict units at new price
    simulated_df['predicted_units'] = model.predict(simulated_df[feature_cols])

    # Compute revenues
    simulated_df['revenue'] = simulated_df['price'] * simulated_df['predicted_units']
    df['revenue'] = df['price'] * df['units']

    # Compute totals
    baseline_revenue = df['revenue'].sum()
    simulated_revenue = simulated_df['revenue'].sum()
    baseline_units = df['units'].sum()
    simulated_units = simulated_df['predicted_units'].sum()
    baseline_price = df['price'].mean()
    simulated_price = simulated_df['price'].mean()

    # Compute changes
    
    pct_change_price = (simulated_price - baseline_price) / baseline_price
    pct_change_units = (simulated_units - baseline_units) / baseline_units

    # Calculate elasticity
    if pct_change_price != 0:
        elasticity = pct_change_units / pct_change_price
    else:
        elasticity = 0

    # Print results
    print(f"Baseline total revenue: {baseline_revenue:,.2f}")
    print(f"Simulated total revenue ({factor_value - 100:+.0f}% on the price): {simulated_revenue:,.2f}")
    print(f"Change in revenue: {((simulated_revenue - baseline_revenue) / baseline_revenue) * 100:.2f}%")
    print(f"Change in price: {pct_change_price * 100:.2f}%")
    print(f"Change in units sold: {pct_change_units * 100:.2f}%")
    print(f"Estimated price elasticity of demand (%Change of Units / %Change of Price): {elasticity:.2f}")


    simulated_df.to_csv(save_path, index=False)