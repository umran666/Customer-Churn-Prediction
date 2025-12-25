# Customer Churn Prediction Using Random Forest

## Project Overview
This project predicts customer churn in a telecom dataset using a Random Forest classifier. The pipeline includes data cleaning, feature encoding, model training, and evaluation with key classification metrics.

## Dataset
- Contains customer demographic information, service subscriptions, billing details, and churn status.
- Target variable: `Churn` (Yes/No).

## Data Preprocessing
- Dropped `customerID` as it is a unique identifier.
- Converted `TotalCharges` column to numeric, handling missing or invalid values.
- Dropped rows with missing data after conversion.
- One-hot encoded categorical features to convert them into numerical format.

## Model Training
- Split dataset into train and test sets with stratification to preserve class distribution.
- Trained a Random Forest classifier with 100 trees and a fixed random seed.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix (visualized)

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
   
2. Run the script:
   ```bash
   customer_churn_predictions.py

3. The script will output evaluation metrics and display the confusion matrix.