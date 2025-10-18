#  Price Elasticity & Revenue Optimization Project

## 1. Project Overview
This project analyzes how product prices and characteristics affect sales (units sold), with the goal of understanding price sensitivity and identifying optimal pricing strategies to maximize total revenue.

### Objectives
- Explore data to understand how price and product features drive sales.
- Build a model predicting units sold from price and product attributes.
- Simulate different pricing strategies to evaluate their impact on revenue.
- Estimate price elasticity of demand.
- Provide business recommendations based on elasticity and simulation results.

## 2. Exploratory Data Analysis
For further data analysis refer to the Data_Anlaysis.ipynb inside the `buynomics/notebooks` folder.

## 3. Data Description
**Dataset:** For running this project, it is needed two datasets, one the product.csv and the other the sales.csv.
Make sure to have attached this datasets into the `buynomics/data` folder.

## 4. Modelling Approach
Target: units
Features: price and encoded product attributes

Models tested:
- Linear Regression
- LightGBM
- XGBoost

## 5. Price simulation & Elasticity
After fitting the model, the project simulates the effect of changing prices on sales and revenue.
Simulation steps:
- Multiply all prices by a chosen inputted factor.
- Predict units at the new price using the previous trained model.
- Compute simulated revenue and compare it with baseline results (Revenue = Price × Predicted Units).

## 6. How to run the code
1. Clone the repository or open the project in your local environment.
2. Install the dependencies needed (requirements.txt) 
```
pip install -r requirements.txt
```
3. Package the code thanks to setup.py. This can be done by running in the main folder of the project:
```
pip install -e .
```
3. Run the `buynomics/src/main/model_selection.py` file to obtain the metrics of each model and generate the pickle file that saves them.
4. Run the `buynomics/src/main/prediction.py` file to predict the units using the saved best model.
5. When prompted, enter a price adjustment (e.g., 95 for -5%).
6. Review printed results and the actual predictions in `buynomics/output/output.csv`, which is generated when executing prediction.py.



Author: Gabriel Galindo
Contact: gabi.galindo02@gmail.com