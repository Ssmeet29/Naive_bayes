import pandas as pd

# Load the dataset
loan_data = pd.read_csv('/kaggle/input/loan-default/Loan_default.csv')  # Replace with the correct path on Kaggle

# Display basic info and first few rows
loan_data.info()
loan_data.head()

# Drop the LoanID column as it's not useful for prediction
loan_data_cleaned = loan_data.drop('LoanID', axis=1)

# Display the updated data to confirm
loan_data_cleaned.head()

from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoders = {}

# Loop through all columns with object data type and apply Label Encoding
for column in loan_data_cleaned.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    loan_data_cleaned[column] = le.fit_transform(loan_data_cleaned[column])
    label_encoders[column] = le  # Store the encoder for later use (if needed)

# Display the first few rows to verify the encoding
loan_data_cleaned.head()

# Split the data into features (X) and target (y)
X = loan_data_cleaned.drop('Default', axis=1)  # Features
y = loan_data_cleaned['Default']  # Target

# Display the shape of the features and target to confirm
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shape of the training and testing sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform the test data
X_test_scaled = scaler.transform(X_test)

# X_train_scaled and X_test_scaled are ready for model training
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train the model on the scaled training data
nb_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = nb_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the evaluation metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Default', 'Default'], 
            yticklabels=['Non-Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

loc="lower right")
plt.show()
