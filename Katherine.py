### Required imports
import pandas as pd
import zipfile
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm  # For detailed regression summary
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

### Function to display regression results for California using statsmodels
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

### Function to display regression results for California using sklearn
def display_california_regression_results_sklearn(df, features, target):
    california_df = df[df['state'] == 'California']
    X_california = california_df[features]
    y_california = california_df[target]

    model = LinearRegression(fit_intercept=True)
    model.fit(X_california, y_california)
    
    y_pred = model.predict(X_california)

    mse = mean_squared_error(y_california, y_pred)
    r2 = r2_score(y_california, y_pred)

    print(f"Regression coefficients (sklearn): {model.coef_}")
    print(f"Intercept (sklearn): {model.intercept_}")
    print(f"Mean Squared Error (sklearn): {mse}")
    print(f"R-squared (sklearn): {r2}")

    california_df['predicted_price'] = y_pred
    print(california_df[['price', 'predicted_price']].head())

### Function to check VIF (Variance Inflation Factor)
def check_vif(df, features):
    X = df[features]
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def plot_relationship(df, feature, target):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature], df[target], alpha=0.5)
    plt.title(f'Scatter plot of {target} vs {feature}')
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.show()


### Main function to orchestrate the process
def main():
    df_cleaned = clean_data(df)

    features = ['bed', 'bath', 'acre_lot', 'house_size']
    target = 'price'

    # Remove properties with bedrooms > 20 (arbitrary threshold based on scatter plot)
    df_cleaned_no_extreme_beds = df_cleaned[df_cleaned['bed'] <= 10]
    # Also remobe bedrooms < 2
    df_cleaned_no_extreme_beds = df_cleaned_no_extreme_beds[df_cleaned_no_extreme_beds['bed'] >= 2]

    # Remove properties with prices > 10 million (arbitrary threshold based on scatter plot)
    df_cleaned_no_extreme_beds_and_prices = df_cleaned_no_extreme_beds[df_cleaned_no_extreme_beds['price'] <= 10**6]

    print(f"Number of observations after removing extreme outliers: {len(df_cleaned_no_extreme_beds_and_prices)}")

    # Re-plot the scatter plot to check if outliers are removed
    plot_relationship(df_cleaned_no_extreme_beds_and_prices, 'bed', 'price')

    # Fit state models without extreme outliers
    state_models = fit_state_models(df_cleaned_no_extreme_beds_and_prices, features, target)
    print(state_models['California']['model'].coef_)

    # print_predicted_prices(state_models)
    
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
    display_california_regression_results_sm(df_cleaned_no_extreme_beds_and_prices, features, target)

    # Display regression results for California using sklearn
    print("\nSklearn Linear Regression Results for California:")
    display_california_regression_results_sklearn(df_cleaned_no_extreme_beds_and_prices, features, target)

    # Example: Plotting price vs bed
    plot_relationship(df_cleaned_no_extreme_beds_and_prices, 'bed', 'price')

if __name__ == "__main__":
    main()
