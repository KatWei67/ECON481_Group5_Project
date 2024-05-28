import pandas as pd
import numpy as np
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
    # Read the CSV file into a pandas DataFrame
    with zip_ref.open(csv_file_name) as csv_file:
        df = pd.read_csv(csv_file)

def clean_data(df):
    # Drop rows with NaNs in specified columns
    df_cleaned = df.dropna(subset=['bed', 'bath', 'acre_lot', 'house_size'])
    return df_cleaned

df_2 = clean_data(df)
