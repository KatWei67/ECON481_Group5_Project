### Required imports
import pandas as pd
import zipfile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

### Function to load data from a zip file
def load_data(zip_file_path, csv_file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        print("Files in zip:", zip_ref.namelist())
        with zip_ref.open(csv_file_name) as csv_file:
            df = pd.read_csv(csv_file)
            print(df.head())
    return df

### Function to clean the data
def clean_data(df):
    df_cleaned = df.copy()
    df_cleaned['status'] = df_cleaned['status'].apply(lambda x: 1 if x == 'for_sale' else 0)
    df_cleaned = df_cleaned[df_cleaned['status'] == 1]

    total_observations = len(df_cleaned)
    print("\nTotal number of observations (rows):", total_observations)
    print("\nMax Missing values in one column:", max(df_cleaned.isnull().sum()))

    df_cleaned.drop(columns=['prev_sold_date'], inplace=True)
    df_cleaned.dropna(inplace=True)
    df_cleaned.drop(columns=['street'], inplace=True)

    total_observations_after = len(df_cleaned)
    print("\nTotal number of observations (rows) After dropping rows with missing values:", total_observations_after)

    print(df_cleaned.head())
    return df_cleaned

### Function to fit Linear Regression models for each state
def fit_state_models(df, features, target):
    state_models = {}
    for state in df['state'].unique():
        print(f"\nFitting Linear Regression model for state: {state}")
        state_df = df[df['state'] == state].copy()
        if len(state_df) < 10:
            print(f"Not enough data for state: {state}")
            continue

        X_state = state_df[features]
        y_state = state_df[target]

        model = LinearRegression()
        model.fit(X_state, y_state)

        state_df['predicted_price'] = model.predict(X_state)

        mse = mean_squared_error(y_state, state_df['predicted_price'])
        r2 = r2_score(y_state, state_df['predicted_price'])

        state_models[state] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'predicted_prices': state_df[['price', 'predicted_price']]
        }

        print(f"Mean Squared Error for state {state}: {mse}")
        print(f"R-squared for state {state}: {r2}")

    return state_models

### Function to print predicted prices for each state
def print_predicted_prices(state_models):
    for state, results in state_models.items():
        print(f"\nPredicted prices for houses in {state}:")
        print(results['predicted_prices'].head())

### Function to predict price based on specific characteristics
def predict_price(state_models, state, bed, bath, acre_lot, house_size):
    if state not in state_models:
        raise ValueError(f"No model found for state: {state}")

    model = state_models[state]['model']
    input_data = pd.DataFrame({
        'bed': [bed],
        'bath': [bath],
        'acre_lot': [acre_lot],
        'house_size': [house_size]
    })

    predicted_price = model.predict(input_data)
    return predicted_price[0]

### Function to plot predicted prices by state
def plot_predicted_prices_by_state(state_models):
    predicted_prices = []
    states = []
    
    for state, results in state_models.items():
        preds = results['predicted_prices']['predicted_price']
        predicted_prices.extend(preds)
        states.extend([state] * len(preds))
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x=states, y=predicted_prices)
    plt.xlabel('State')
    plt.ylabel('Predicted Price')
    plt.title('Predicted Prices by State')
    plt.xticks(rotation=90)
    plt.show()

### Function to plot model performance by state
def plot_model_performance(state_models):
    states = []
    mses = []
    r2s = []
    
    for state, metrics in state_models.items():
        states.append(state)
        mses.append(metrics['mse'])
        r2s.append(metrics['r2'])
    
    plt.figure(figsize=(16, 10))
    
    # MSE bar chart
    plt.subplot(2, 1, 1)
    sns.barplot(x=states, y=mses)
    plt.ylabel('Mean Squared Error')
    plt.title('Model Performance by State - MSE')
    plt.xticks(rotation=45, ha='right')  # Rotate state names and align right
    
    # R² bar chart
    plt.subplot(2, 1, 2)
    sns.barplot(x=states, y=r2s)
    plt.ylabel('R-squared')
    plt.title('Model Performance by State - R²')
    plt.xticks(rotation=45, ha='right')  # Rotate state names and align right
    
    plt.tight_layout()
    plt.show()

### Main function to orchestrate the process
def main():
    zip_file_path = 'usa-real-estate-dataset.zip'
    csv_file_name = 'realtor-data.zip.csv'

    df = load_data(zip_file_path, csv_file_name)
    df_cleaned = clean_data(df)

    features = ['bed', 'bath', 'acre_lot', 'house_size']
    target = 'price'

    state_models = fit_state_models(df_cleaned, features, target)
    print_predicted_prices(state_models)
    
    # show three examples of how predict model works
    predicted_price = predict_price(state_models, 'California', 3, 2, 0.25, 2000)
    print(f"Predicted price for the house in California: ${predicted_price:.2f}")
    predicted_price = predict_price(state_models, 'Texas', 4, 3, 0.5, 3000)
    print(f"Predicted price for the house in Texas: ${predicted_price:.2f}")
    predicted_price = predict_price(state_models, 'Florida', 2, 1, 0.1, 1500)
    print(f"Predicted price for the house in Florida: ${predicted_price:.2f}")

    # visualization 
    plot_predicted_prices_by_state(state_models)
    plot_model_performance(state_models)

if __name__ == "__main__":
    main()
