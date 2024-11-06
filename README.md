# Stroke Prediction Project

This project analyzes and predicts the likelihood of stroke occurrence using various machine learning algorithms. It leverages Apache Spark for distributed computing, enabling efficient data processing and model training across multiple virtual machines (VMs). The analysis focuses on data preprocessing, handling imbalanced data, and evaluating different machine learning models, including K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Random Forest, and XGBoost, to identify the best-performing model.

## Project Objectives

1. **Data Preprocessing**: Clean and transform the data, handle missing values, apply feature scaling, and manage class imbalance.
2. **Model Evaluation**: Train and evaluate various machine learning models for classification and compare their performance in terms of accuracy, AUC, recall, and precision.
3. **Distributed Computing**: Utilize Apache Spark to run the analysis across multiple VMs, enhancing computational speed and efficiency.
4. **Performance Comparison**: Analyze the impact of using different numbers of VMs on computational performance.

## Dataset

The dataset used in this project contains over 5,000 records with features related to patient health metrics and lifestyle, such as age, hypertension, heart disease, average glucose level, and BMI. The target variable indicates whether a patient has experienced a stroke.

**Data Source**: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Data Preprocessing Steps

1. **Dropped Unnecessary Columns**: Removed the 'id' column as it does not contribute to the analysis.
2. **Converted Data Types**: Converted categorical variables like hypertension, heart_disease, and stroke to string types for consistency.
3. **Handled Missing Values**: Used KNN imputation to fill missing values in the BMI column.
4. **Outlier Treatment**: Applied log transformation to the avg_glucose_level and bmi columns to reduce the impact of outliers.
5. **Encoding**: Converted categorical variables to numerical format using one-hot encoding.
6. **Standardization**: Scaled numerical columns to have a mean of 0 and a standard deviation of 1.
7. **Addressed Class Imbalance**: Used oversampling to balance the dataset, ensuring equal representation of stroke and non-stroke cases.

## Machine Learning Models

The following machine learning models were trained and evaluated for stroke prediction:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Classifier (SVM)**
- **Random Forest**
- **XGBoost**

Each model was assessed based on its accuracy, AUC, recall, and precision.

## Results

The models performances were compared, and the Random Forest model showed the best accuracy and AUC scores. Below is a summary of the model performance:

| Model           | Accuracy | AUC   | Precision | Recall |
|-----------------|----------|-------|-----------|--------|
| KNN             | 0.9738   | 0.9739| 1.00      | 0.95   |
| SVM             | 0.8277   | 0.8808| 0.87      | 0.75   |
| Random Forest   | 0.9938   | 1.0000| 1.00      | 0.98   |
| XGBoost         | 0.9779   | 0.9985| 1.00      | 0.96   |

## Performance Comparison

To measure computational efficiency, the program was run under different VM setups:

1. **Single VM**: Model training and evaluation took a longer time due to limited resources.
2. **Two VMs**: Performance improved, and the computational speed increased as Spark distributed tasks across the VMs.

## Python Packages Used

- **Pandas**: Data handling and manipulation
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **XGBoost**: Gradient boosting classifier
- **Imbalanced-learn**: Handling imbalanced datasets


