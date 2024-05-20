### What is the approximate selling price of a house given some specific characteristics?
### Read dataset
import zipfile
import pandas as pd
import statsmodels.api as sm

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

### Data clean
# 1. Inspect Data
# Check for missing values
# Total number of observations (rows)
total_observations = len(df)
print("\nTotal number of observations (rows):", total_observations)
print("\nMax Missing values in one column:")
print(max(df.isnull().sum()))

# Drop the row with empty value
# Drop 'prev_sold_date' column due to excessive missing values
df.drop(columns=['prev_sold_date'], inplace=True)

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# Drop 'street' since it is not relevant
df.drop(columns=['street'], inplace=True)

# Check total number of observations after data clean
total_observations = len(df)
print("\nTotal number of observations (rows) After drop missing value row:", total_observations)

print(df.head())

# Handle categorical variables
df = pd.get_dummies(df, columns=['city', 'state'], drop_first=True)

# Define the dependent variable (target)
Y = df['price']

# Define the independent variables (features)
X = df.drop(columns=['price'])  # Exclude the target column

# Add a constant to the independent variables matrix (for the intercept term)
X = sm.add_constant(X)

# Build and fit the OLS model
model = sm.OLS(Y, X).fit()

# Print the model summary
print("\nOLS Model Summary:")
print(model.summary())

# Predict the prices for each city/state combination
# Get unique combinations of city and state
unique_combinations = df.drop(columns=['price']).drop_duplicates()

# Add a constant to these combinations
unique_combinations = sm.add_constant(unique_combinations)

# Predict prices
unique_combinations['predicted_price'] = model.predict(unique_combinations)

# Extract city and state labels for the unique combinations
# Create a reverse mapping from dummy variables back to original labels
city_state_mapping = df[['city', 'state']].drop_duplicates().reset_index(drop=True)
unique_combinations_with_labels = pd.concat([city_state_mapping, unique_combinations.reset_index(drop=True)], axis=1)

# Display the predictions
print("\nPredicted Prices for Each City/State Combination:")
print(unique_combinations_with_labels[['city', 'state', 'predicted_price']])