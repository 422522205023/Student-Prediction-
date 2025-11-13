import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np # Import numpy for rounding function
# --- 1. Load the Dataset ---
file_path = 'student-mat.csv'
df = pd.read_csv(file_path)
# --- 2. Define Features (X) and Target (y) ---
features = ['Dalc', 'Walc', 'G1', 'G2']
X = df[features]
y = df['G3']
# --- 3. Split the Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# --- 4. Train the Model ---
model = LinearRegression()
model.fit(X_train, y_train)
# --- 5. Make Predictions ---
y_pred = model.predict(X_test)
# --- 6. Round Predictions to the Nearest Integer (Whole Number) ---
# This step converts the predicted grades from decimals to integers
predicted_g3_rounded = np.round(y_pred).astype(int)
# Optional: Ensure grades are within the expected 0-20 range (important for grades)
# This clips any predicted value that is outside the [0, 20] range.
predicted_g3_final = np.clip(predicted_g3_rounded, 0, 20)
predicted_g3_rounded = np.round(y_pred).astype(int)
predicted_g3_final = np.clip(predicted_g3_rounded, 0, 20)
# --- 7. Create the Output CSV File ---
# Combine the test data (features, actual G3, and rounded predicted G3)
X_test_reset = X_test.copy()
X_test_reset['Actual_G3'] = y_test
X_test_reset = X_test_reset.reset_index(drop=True)
# Create a DataFrame for the final rounded predictions
predictions_df = pd.DataFrame(
    {'Predicted_G3_Rounded': predicted_g3_final}
)
# Merge the test data and the predictions
output_df = pd.concat([X_test_reset, predictions_df], axis=1)
# Save the combined DataFrame to a new CSV file
output_file_name = 'student_grade_predictions_rounded.csv'
output_df.to_csv(output_file_name, index=False)
# --- 8. Print Confirmation and Evaluation ---
# We calculate evaluation metrics using the original floating-point predictions
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Prediction and Rounding completed successfully.")
print(f"The results (Predicted G3 as whole numbers) have been saved to: **{output_file_name}**")
print("-" * 50)
print("Model Evaluation (using original decimal predictions):")
print(f"R-squared (R²) Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
