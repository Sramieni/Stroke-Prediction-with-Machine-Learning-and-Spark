# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Set visualization settings for seaborn
sns.set_context('notebook', font_scale=1.2)

# Step 1: Load the dataset
data = pd.read_csv('/home/Sat5165/Downloads/stroke_data.csv')
print("Initial Data Preview:")
print(data.head())

# Step 2: Drop the 'id' column, as it does not contribute to the analysis
data.drop('id', axis=1, inplace=True)
print("\nData after dropping 'id' column:")
print(data.head())

# Step 3: Convert specific columns to categorical types for consistency
# Here, 'hypertension', 'heart_disease', and 'stroke' are treated as strings (categorical)
data[['hypertension', 'heart_disease', 'stroke']] = data[['hypertension', 'heart_disease', 'stroke']].astype(str)
print("\nData types after conversion to categorical:")
print(data.dtypes)

# Step 4: Apply log transformation to 'avg_glucose_level' and 'bmi' columns to handle outliers
data[['avg_glucose_level', 'bmi']] = data[['avg_glucose_level', 'bmi']].apply(np.log)
print("\nData after log transformation on 'avg_glucose_level' and 'bmi':")
print(data[['avg_glucose_level', 'bmi']].describe())

# Step 5: Remove any instances with 'Other' in the 'gender' column for simplicity
data = data[data['gender'] != 'Other']
print("\nData after removing rows with 'Other' in 'gender':")
print(data['gender'].value_counts())

# Step 6: Fill missing values in 'bmi' column using KNN imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
data[['bmi']] = imputer.fit_transform(data[['bmi']])
print("\nData after KNN imputation on 'bmi' (checking for missing values):")
print(data.isnull().sum())

# Step 7: One-hot encode categorical columns to convert them into numerical values
data = pd.get_dummies(data, drop_first=True)
print("\nData after one-hot encoding categorical variables:")
print(data.head())

# Step 8: Standardize numerical columns to ensure mean=0 and std=1
scaler = StandardScaler()
data[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(data[['age', 'avg_glucose_level', 'bmi']])
print("\nData after standardizing numerical columns ('age', 'avg_glucose_level', 'bmi'):")
print(data[['age', 'avg_glucose_level', 'bmi']].describe())

# Step 9: Separate features (X) and target variable (y)
X = data.drop('stroke_1', axis=1)
y = data['stroke_1']

# Step 10: Address class imbalance by oversampling the minority class (stroke cases)
oversample = RandomOverSampler(sampling_strategy='minority')
X, y = oversample.fit_resample(X, y)
print("\nData after addressing class imbalance (oversampling):")
print(y.value_counts())

# Step 11: Split data into training and testing sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models and print results with a heatmap for the confusion matrix
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Predict probabilities for AUC calculation
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Print accuracy, AUC, and classification report
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC AUC Score: {auc:.4f}')
    print(classification_report(y_test, y_pred))
    
    # Display the confusion matrix as a heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False, 
                xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.show()

# Train and evaluate KNN model
print("K-Nearest Neighbors (KNN) Results:")
knn = KNeighborsClassifier(n_neighbors=2)
evaluate_model(knn, X_train, X_test, y_train, y_test)

# Train and evaluate Support Vector Classifier (SVM)
print("\nSupport Vector Classifier (SVM) Results:")
svm = SVC(C=1, gamma=0.1, probability=True)
evaluate_model(svm, X_train, X_test, y_train, y_test)

# Train and evaluate Random Forest model
print("\nRandom Forest Results:")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf, X_train, X_test, y_train, y_test)

# Train and evaluate XGBoost model
print("\nXGBoost Results:")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
evaluate_model(xgb, X_train, X_test, y_train, y_test)

# Bar plot to visualize the accuracy scores of each model
models = ['KNN', 'SVM', 'Random Forest', 'XGBoost']
accuracy_scores = [
    accuracy_score(y_test, knn.predict(X_test)),
    accuracy_score(y_test, svm.predict(X_test)),
    accuracy_score(y_test, rf.predict(X_test)),
    accuracy_score(y_test, xgb.predict(X_test))
]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracy_scores, palette='viridis')
plt.ylabel('Accuracy Score')
plt.title('Model Performance Comparison')
plt.show()
