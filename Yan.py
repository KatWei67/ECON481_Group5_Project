import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


### Load Data
# Define the path to the zip file and the name of the CSV file within the zip
zip_file_path = 'usa-real-estate-dataset.zip'
csv_file_name = 'realtor-data.zip.csv'  # Adjust if the file name inside the zip is different

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Read the CSV file into a pandas DataFrame
    with zip_ref.open(csv_file_name) as csv_file:
        df = pd.read_csv(csv_file)

def clean_data(df):
    # Drop rows with NaNs in specified columns
    df_cleaned = df.dropna(subset=['bed', 'bath', 'acre_lot', 'house_size', 'price'])
    return df_cleaned

# Example usage
df_2 = clean_data(df)

#seed
seed = 481
np.random.seed(seed)

# Sample 10% of the rows since dataset is big
#df_2 = df_2.sample(frac=0.1, random_state=1)
def find_houses(data, budget=None, min_bedrooms=0, min_bathrooms=0, preferred_state=None, preferred_city=None, min_lot_size=0, max_lot_size=None):
    # Filter based on budget if provided
    if budget is not None:
        filtered_data = data[(data['price'] <= budget) & (data['status'] == 'for_sale')]
    else:
        filtered_data = data[data['status'] == 'for_sale']

    # Filter based on number of bedrooms
    if min_bedrooms > 0:
        filtered_data = filtered_data[filtered_data['bed'] >= min_bedrooms]

    # Filter based on number of bathrooms
    if min_bathrooms > 0:
        filtered_data = filtered_data[filtered_data['bath'] >= min_bathrooms]

    # Filter based on preferred state
    if preferred_state:
        filtered_data = filtered_data[filtered_data['state'].str.lower() == preferred_state.lower()]

    # Filter based on preferred city
    if preferred_city:
        filtered_data = filtered_data[filtered_data['city'].str.lower() == preferred_city.lower()]

    # Filter based on lot size
    if min_lot_size > 0:
        filtered_data = filtered_data[filtered_data['acre_lot'] >= min_lot_size]
        
    if max_lot_size is not None:
        filtered_data = filtered_data[filtered_data['acre_lot'] <= max_lot_size]

    return filtered_data

# Example usage
budget = 1000000
min_bedrooms = 3
min_bathrooms = 2
preferred_state = 'Washington'
preferred_city = 'Seattle'
min_lot_size = 0.1
max_lot_size = 5.0

results = find_houses(df_2, budget, min_bedrooms, min_bathrooms, preferred_state, preferred_city)
print(results.head())

min_bedrooms = 1
min_lot_size = 0.2
preferred_state = 'Colorado'

results = find_houses(df_2, min_bedrooms = min_bedrooms, min_lot_size = 0.2, preferred_state = preferred_state)
print(results.head())

def plot_house_frequency(df, budget=None, min_bedrooms=0, min_bathrooms=0, preferred_state=None, preferred_city=None, min_lot_size=0, max_lot_size=None):
    # Call the find_houses function to filter the data
    filtered_data = find_houses(df, budget, min_bedrooms, min_bathrooms, preferred_state, preferred_city, min_lot_size, max_lot_size)

    if preferred_city:
        # Plot frequency by street if city is specified
        plot_data = filtered_data['street'].value_counts()
        title = f'House Frequency by Street in {preferred_city}'
        xlabel = 'Street'
    elif preferred_state:
        # Plot frequency by city if state is specified but city is not
        plot_data = filtered_data['city'].value_counts()
        title = f'House Frequency by City in {preferred_state}'
        xlabel = 'City'
    else:
        # Plot frequency by state if state is not specified
        plot_data = filtered_data['state'].value_counts()
        title = 'House Frequency by State'
        xlabel = 'State'

    # Plotting the data
    plt.figure(figsize=(14, 7))
    plot_data.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.show()

budget = 200000
min_bedrooms = 3
min_bathrooms = 2
preferred_state = 'Washington'
preferred_city = None
min_lot_size = 0.1
max_lot_size = 1.0

plot_house_frequency(df_2, budget, min_bedrooms, min_bathrooms, preferred_state, preferred_city, min_lot_size, max_lot_size)

budget = 100000
min_bedrooms = 6
min_bathrooms = 4

plot_house_frequency(df_2, budget, min_bedrooms, min_bathrooms)

min_bedrooms = 8
min_bathrooms = 4
city = 'Seattle'
plot_house_frequency(df_2, min_bedrooms = min_bedrooms, min_bathrooms = min_bathrooms, preferred_city = city)

# Define the feature columns and the target column
feature_columns = ['price', 'bed', 'bath', 'acre_lot', 'house_size']
target_column = 'state'  # or 'zip_code'

# Drop rows with missing target values
df_2 = df_2.dropna(subset=[target_column])

# Prepare the feature matrix (X) and the target vector (y)
X = df_2[feature_columns]
y = df_2[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with a StandardScaler and Logistic Regression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)
