import pandas as pd
# Load the dataset
url = 'Iris.csv'
iris_data = pd.read_csv(url)

# Display the first few rows of the dataset
print(iris_data.head())
# Check for missing values
print(iris_data.isnull().sum())

# Encode species into numerical values if needed
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
iris_data['Species'] = label_encoder.fit_transform(iris_data['Species'])
from sklearn.model_selection import train_test_split

# Define features and labels
X = iris_data.drop('Species', axis=1)  # Features
y = iris_data['Species']                 # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

# Initialize the model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

#make predictions
predictions=model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print(classification_report(y_test, predictions))
import matplotlib.pyplot as plt
import seaborn as sns

# Create a confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()