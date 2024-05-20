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

### Data cleaning
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