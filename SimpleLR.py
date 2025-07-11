# Install required libraries if needed
# pip install pandas scikit-learn matplotlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Path to the CSV file on your desktop
# (Change the username if needed)
file_path = os.path.expanduser("~/Desktop/ML Projects/Salary_dataset.csv")

# Load the CSV file
df = pd.read_csv(file_path)

# Show first few records
print("First 5 records:\n", df.head())

# Separate features (X) and target (y)
X = df[["YearsExperience"]]
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Coefficient (slope):", model.coef_[0])
print("Model Intercept (bias):", model.intercept_)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plotting the regression line on test data
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience (Test Set)")
plt.legend()
plt.grid(True)
plt.show()
