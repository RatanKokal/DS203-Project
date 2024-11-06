import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the model

# Step 1: Load the dataset
df = pd.read_csv('database.csv')  # Replace with your dataset path

# Step 2: Split features and target
X = df.drop(columns=['Class'])  # Features (all columns except 'Class')
y = df['Class']  # Target (the 'Class' column)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize the features (important for SVM to perform well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Initialize and train the SVM classifier
svm_classifier = SVC(kernel='rbf', random_state=42)  # You can change the kernel as needed
svm_classifier.fit(X_train_scaled, y_train)

# Step 6: Save the model and scaler
joblib.dump(svm_classifier, 'svm_model.pkl')  # Save the trained SVM model
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

# Step 7: Predict on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Step 8: Evaluate the model
accuracy_train = svm_classifier.score(X_train_scaled, y_train)  # Calculate training accuracy
accuracy_test = accuracy_score(y_test, y_pred)  # Calculate test accuracy

print(f'Training Accuracy: {accuracy_train:.4f}')
print(f'Test Accuracy: {accuracy_test:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
