### What is the approximate selling price of a house given some specific characteristics?
### Read dataset
import zipfile
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


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

### Data cleaning

# Convert 'status' column to dummy variable: 1 for 'for_sale', 0 for other statuses
df['status'] = df['status'].apply(lambda x: 1 if x == 'for_sale' else 0)

# Filter dataset to include only 'for_sale' houses
df = df[df['status'] == 1]

# Inspect Data
# Check for missing values
# Total number of observations (rows)
total_observations = len(df)
print("\nTotal number of observations (rows):", total_observations)
print("\nMax Missing values in one column:")
print(max(df.isnull().sum()))

# Drop 'prev_sold_date' column due to excessive missing values
df.drop(columns=['prev_sold_date'], inplace=True)

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# Drop 'street' since it is not relevant
df.drop(columns=['street'], inplace=True)

# Check total number of observations after data clean
total_observations = len(df)
print("\nTotal number of observations (rows) After dropping rows with missing values:", total_observations)

print(df.head())

# Handle any new missing values resulting from conversion to numeric
df.dropna(inplace=True)

# Sample 20% of the data
df_sampled = df.sample(frac=1, random_state=42)

# Define the features and target variable
features = ['bed', 'bath', 'acre_lot', 'house_size']
target = 'price'

# Create a dictionary to store the models and results for each state
state_models = {}

# Loop through each unique state and fit a Linear Regression model
for state in df_sampled['state'].unique():
    print(f"\nFitting Linear Regression model for state: {state}")
    
    # Filter the data for the current state
    state_df = df_sampled[df_sampled['state'] == state].copy()
    
    # Check if there are enough observations for this state
    if len(state_df) < 10:  # Arbitrary threshold to ensure sufficient data
        print(f"Not enough data for state: {state}")
        continue
    
    # Define features (X) and target (y)
    X_state = state_df[features]
    y_state = state_df[target]
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Fit the model
    model.fit(X_state, y_state)
    
    # Predict prices for the state
    state_df['predicted_price'] = model.predict(X_state)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_state, state_df['predicted_price'])
    r2 = r2_score(y_state, state_df['predicted_price'])
    
    # Store the model and performance metrics in the dictionary
    state_models[state] = {
        'model': model,
        'mse': mse,
        'r2': r2,
        'predicted_prices': state_df[['price', 'predicted_price']]
    }
    
    # Print the performance metrics
    print(f"Mean Squared Error for state {state}: {mse}")
    print(f"R-squared for state {state}: {r2}")

# Example usage: Show predicted prices for each state
for state, results in state_models.items():
    print(f"\nPredicted prices for houses in {state}:")
    print(results['predicted_prices'].head())


# Function to predict price based on specific characteristics
def predict_price(state, bed, bath, acre_lot, house_size):
    if state not in state_models:
        raise ValueError(f"No model found for state: {state}")
    
    model = state_models[state]['model']
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'bed': [bed],
        'bath': [bath],
        'acre_lot': [acre_lot],
        'house_size': [house_size]
    })
    
    # Predict the price
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Example usage:
# Predict the price of a house in California with 3 beds, 2 baths, 0.25 acre_lot, and 2000 house_size
predicted_price = predict_price('California', 3, 2, 0.25, 2000)
print(f"Predicted price for the house in California: ${predicted_price:.2f}")

# Show predicted prices for each state
for state, results in state_models.items():
    print(f"\nPredicted prices for houses in {state}:")
    print(results['predicted_prices'].head())

### Visualizations

# Box plot of predicted prices for each state
def plot_predicted_prices_by_state():
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

# Example usage:
plot_predicted_prices_by_state()

# Bar chart of model performance by state with better label formatting
def plot_model_performance():
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

# Example usage:
plot_model_performance()
