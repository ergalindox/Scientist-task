
import copy
from buynomics.src.utils.functions import train_data


def prediction(model, df, save_path):

    X, _ = train_data(df)
    simulated_df = copy.deepcopy(X)
    feature_cols = list(X.columns)
    factor_value = float(input("Enter desired price level as a percentage (e.g., 120 = +20%, 80 = -20%): "))
    simulated_df['price'] = simulated_df['price'] * (factor_value/100)
    simulated_df['predicted_units'] = model.predict(simulated_df[feature_cols])

    # Compute Revenue
    simulated_df['revenue'] = simulated_df['price'] * simulated_df['predicted_units']
    df['revenue'] = df['price'] * df['units']

    baseline_revenue = df['revenue'].sum()
    simulated_revenue = simulated_df['revenue'].sum()

    print(f"Baseline total revenue: {baseline_revenue:,.2f}")
    print(f"Simulated total revenue ({factor_value - 100:+.0f}% on the price): {simulated_revenue:,.2f}")
    print(f"Change in revenue: {((simulated_revenue - baseline_revenue) / baseline_revenue) * 100:.2f}%")


    simulated_df.to_csv(save_path, index=False)