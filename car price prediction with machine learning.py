import pandas as pd

# Load the dataset
file_path = 'car data.csv' 
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())
# Check the data types and for missing values
print(data.info())
print(data.isnull().sum())

# Get descriptive statistics
print(data.describe())

# Example: Dropping rows with missing target values (price)
data.dropna(subset=['Selling_Price'], inplace=True)

# Fill missing values in other columns if necessary
data.fillna(method='ffill', inplace=True)  
# Example: Selecting features and target variable
X = data[['Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission']]  # Replace with your actual column names
y = data['Selling_Price']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line of equality
plt.show()