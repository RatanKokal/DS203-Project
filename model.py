import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib  # For saving models

# Step 1: Load the dataset
df = pd.read_csv('database.csv')  # Replace with your dataset path

# Step 2: Split features and target
X = df.drop(columns=['Class'])  # Features (all columns except 'Class')
y = df['Class']  # Target (the 'Class' column)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Step 4: Standardize the features (important for many classifiers to perform well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Initialize classifiers
classifiers = {
    'SVM (RBF Kernel)': SVC(kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Step 6: Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    joblib.dump(clf, f'{clf_name.lower().replace(" ", "_")}_model_1.pkl')  # Save the trained model
    
    # Predict on the test set
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy_train = clf.score(X_train_scaled, y_train)  # Training accuracy
    accuracy_test = accuracy_score(y_test, y_pred)  # Test accuracy
    
    # Print results
    print(f'Classifier: {clf_name}')
    print(f'Training Accuracy: {accuracy_train:.4f}')
    print(f'Test Accuracy: {accuracy_test:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print('-' * 60)

# Step 7: Save the scaler (common for all classifiers)
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
