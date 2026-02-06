# BillGuard-AI
# Fake Bill Detection System using Python
# This system uses machine learning to classify bills as Real or Fake based on given features.
# We'll use Random Forest as the ML algorithm for classification.
# Libraries: Pandas for data handling, NumPy for numerical operations, Scikit-learn for ML.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Step 1: Create a sample dataset (In a real scenario, load from a CSV file using pd.read_csv('bills.csv'))
# Dataset fields: Bill ID, Hospital/Vendor Name, Bill Amount, Date, Item Description, Quantity, Total Amount, Label
data = {
    'Bill ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Hospital/Vendor Name': ['Hospital A', 'Vendor B', 'Hospital C', 'Vendor D', 'Hospital A', 'Vendor B', 'Hospital C', 'Vendor D', 'Hospital A', 'Vendor B'],
    'Bill Amount': [100, 200, 150, 300, 120, 250, 180, 350, 110, 220],
    'Date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01'],
    'Item Description': ['Medicine', 'Equipment', 'Surgery', 'Supplies', 'Medicine', 'Equipment', 'Surgery', 'Supplies', 'Medicine', 'Equipment'],
    'Quantity': [1, 2, 1, 3, 1, 2, 1, 3, 1, 2],
    'Total Amount': [100, 400, 150, 900, 120, 500, 180, 1050, 110, 440],
    'Label': ['Real', 'Fake', 'Real', 'Fake', 'Real', 'Fake', 'Real', 'Fake', 'Real', 'Fake']
}

df = pd.DataFrame(data)
print("Sample Dataset:")
print(df)
print("\n")

# Step 2: Data Cleaning and Preprocessing
# - Handle missing values (if any, fill with mean for numerical, mode for categorical)
# - Convert Date to datetime and extract features like month and day
# - Encode categorical variables (Hospital/Vendor Name, Item Description) using LabelEncoder
# - Encode Label: Real=1, Fake=0

# Check for missing values
print("Checking for missing values:")
print(df.isnull().sum())
print("\n")

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Drop original Date column as we extracted features
df = df.drop('Date', axis=1)

# Encode categorical columns
le = LabelEncoder()
df['Hospital/Vendor Name'] = le.fit_transform(df['Hospital/Vendor Name'])
df['Item Description'] = le.fit_transform(df['Item Description'])

# Encode Label
df['Label'] = df['Label'].map({'Real': 1, 'Fake': 0})

print("Preprocessed Dataset:")
print(df)
print("\n")

# Step 3: Prepare features (X) and target (y)
# Features: All columns except 'Label' and 'Bill ID' (Bill ID is unique, not useful for prediction)
X = df.drop(['Bill ID', 'Label'], axis=1)
y = df['Label']

print("Features (X):")
print(X.head())
print("Target (y):")
print(y.head())
print("\n")

# Step 4: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train-Test Split completed. Train size:", len(X_train), "Test size:", len(X_test))
print("\n")

# Step 5: Apply Machine Learning Algorithm - Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model trained using Random Forest.")
print("\n")

# Step 6: Predict on test set and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\n")

# Step 7: Allow user to input a new bill and predict
# Function to preprocess user input similarly
def preprocess_input(hospital_vendor, bill_amount, item_desc, quantity, total_amount, month, day):
    # Encode categorical inputs using the same LabelEncoder
    hospital_encoded = le.transform([hospital_vendor])[0]
    item_encoded = le.transform([item_desc])[0]
    return [hospital_encoded, bill_amount, item_encoded, quantity, total_amount, month, day]

print("Enter details for a new bill to predict if it's Real or Fake:")
hospital_vendor = input("Hospital/Vendor Name (e.g., Hospital A): ")
bill_amount = float(input("Bill Amount (e.g., 100): "))
item_desc = input("Item Description (e.g., Medicine): ")
quantity = int(input("Quantity (e.g., 1): "))
total_amount = float(input("Total Amount (e.g., 100): "))
date_str = input("Date (YYYY-MM-DD, e.g., 2023-01-01): ")

# Extract month and day from date
date_obj = datetime.strptime(date_str, '%Y-%m-%d')
month = date_obj.month
day = date_obj.day

# Preprocess input
input_features = preprocess_input(hospital_vendor, bill_amount, item_desc, quantity, total_amount, month, day)
input_df = pd.DataFrame([input_features], columns=X.columns)

# Predict
prediction = model.predict(input_df)[0]
result = "Real" if prediction == 1 else "Fake"
print(f"\nPrediction: The bill is {result}.")
