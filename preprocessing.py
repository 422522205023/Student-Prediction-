import pandas as pd
# Define the columns used in your model
model_columns = ['Dalc', 'Walc', 'G1', 'G2', 'G3']
# Load the dataset
file_path = 'student-mat.csv'
df = pd.read_csv(file_path)
# Subset the DataFrame
df_subset = df[model_columns]
# Check for missing values in the subset
missing_values = df_subset.isnull().sum()
# Print only columns that have missing values (count > 0)
print("Missing Value Count in Subset:")
print(missing_values[missing_values > 0])
print(f"Total columns with missing values: {len(missing_values[missing_values > 0])}")
