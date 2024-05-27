### Required imports
import pandas as pd
import zipfile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm  # For detailed regression summary

### Load Data
# Define the path to the zip file and the name of the CSV file within the zip
zip_file_path = 'usa-real-estate-dataset.zip'
csv_file_name = 'realtor-data.zip.csv'  # Adjust if the file name inside the zip is different

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # List all files in the zip archive (optional)
    print("Files in zip:", zip_ref.namelist())

    # Read the CSV file into a pandas DataFrame
    with zip_ref.open(csv_file_name) as csv_file:
        df = pd.read_csv(csv_file)
        print(df.head())  # Display the first few rows of the DataFrame

### Function to clean the data
def clean_data(df):
    df_cleaned = df.copy()
    df_cleaned['status'] = df_cleaned['status'].apply(lambda x: 1 if x == 'for_sale' else 0)
    df_cleaned = df_cleaned[df_cleaned['status'] == 1]

    df_cleaned.drop(columns=['prev_sold_date', 'street'], inplace=True)
    df_cleaned.dropna(inplace=True)

    return df_cleaned

df_cleaned = clean_data(df)

# Function to fit Linear Regression models for each state
def fit_state_models(df, features, target):
    state_models = {}
    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        if len(state_df) < 10:
            continue

        X_state = state_df[features]
        y_state = state_df[target]

        model = LinearRegression(fit_intercept=True)
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

    return state_models

features = ['bed', 'bath', 'acre_lot', 'house_size']
target = 'price'
state_models = fit_state_models(df_cleaned, features, target)

# Function to plot predicted prices by state
def plot_predicted_prices_by_state(state_models):
    predicted_prices = []
    states = []
    
    for state, results in state_models.items():
        preds = results['predicted_prices']['predicted_price']
        # Filter out values less than zero
        preds = preds[preds >= 0]
        predicted_prices.extend(preds)
        states.extend([state] * len(preds))
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x=states, y=predicted_prices)
    plt.xlabel('State')
    plt.ylabel('Predicted Price')
    plt.title('Predicted Prices by State')
    plt.xticks(rotation=90)
    plt.show()

plot_predicted_prices_by_state(state_models)

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

# Function to predict price based on specific characteristics
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

# Function to display regression results for California using statsmodels
def display_california_regression_results_sm(df, features, target):
    california_df = df[df['state'] == 'California']
    X_california = california_df[features]
    y_california = california_df[target]

    # Add a constant to the independent variables matrix (for the intercept)
    X_california = sm.add_constant(X_california)

    # Fit the regression model
    model = sm.OLS(y_california, X_california).fit()

    # Print the regression results
    print(model.summary())

# Main function to orchestrate the process
def main():
    df_cleaned = clean_data(df)

    features = ['bed', 'bath', 'acre_lot', 'house_size']
    target = 'price'

    # Remove properties with extreme values based on provided criteria
    df_cleaned_no_extreme_data = df_cleaned[df_cleaned['bed'] <= 10]
    df_cleaned_no_extreme_data = df_cleaned_no_extreme_data[df_cleaned_no_extreme_data['bed'] >= 2]
    df_cleaned_no_extreme_data = df_cleaned_no_extreme_data[df_cleaned_no_extreme_data['bath'] <= 10]
    df_cleaned_no_extreme_data = df_cleaned_no_extreme_data[df_cleaned_no_extreme_data['acre_lot'] <= 500]
    df_cleaned_no_extreme_data = df_cleaned_no_extreme_data[df_cleaned_no_extreme_data['house_size'] <= 15000]
    df_cleaned_no_extreme_data = df_cleaned_no_extreme_data[df_cleaned_no_extreme_data['price'] <= 10**6]

    print(f"Number of observations after removing extreme outliers: {len(df_cleaned_no_extreme_data)}")

    # Fit state models without extreme outliers
    state_models = fit_state_models(df_cleaned_no_extreme_data, features, target)
    print(state_models['California']['model'].coef_)

    # Show three examples of how the predict model works
    predicted_price = predict_price(state_models, 'California', 3, 2, 0.25, 2000)
    print(f"Predicted price for the house in California: ${predicted_price:.2f}")
    predicted_price = predict_price(state_models, 'Texas', 4, 3, 0.5, 3000)
    print(f"Predicted price for the house in Texas: ${predicted_price:.2f}")
    predicted_price = predict_price(state_models, 'Florida', 2, 1, 0.1, 1500)
    print(f"Predicted price for the house in Florida: ${predicted_price:.2f}")

    # Visualization 
    plot_predicted_prices_by_state(state_models)
    plot_model_performance(state_models)

    # Display regression results for California using statsmodels
    print("\nStatsmodels OLS Regression Results for California:")
    display_california_regression_results_sm(df_cleaned_no_extreme_data, features, target)

if __name__ == "__main__":
    main()
