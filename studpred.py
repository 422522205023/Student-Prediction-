import pandas as pd

# Define the columns used in your model
model_columns = ['Dalc', 'Walc', 'G1', 'G2', 'G3']
# Load the dataset
file_path = 'student-mat.csv'
df = pd.read_csv(file_path)
# Subset the DataFrame
df_subset = df[model_columns]
# Print the head of the subset
print("Head of the selected columns:")
print(df_subset.head())
