# Livestock_disease_prediction

 A Machine Learning Approach using SVM & Logistic Regression

This repository contains a Livestock Disease Prediction Model built using Support Vector Machine (SVM) and Logistic Regression, achieving 24% and 25% accuracy, respectively. The model helps farmers and veterinarians detect potential diseases in livestock based on symptoms, age, and temperature readings.

Model Performance
Model	- Accuracy (%) :
   -Logistic Regression	 - 25%
   -Support Vector Machine (SVM) -	24%
🔹 Note: While accuracy is currently modest, the model serves as a foundation for future improvements with better datasets and feature engineering.


 Predicts diseases based on:
   - Animal type (Cow, Buffalo, Sheep, Goat) ,Age ,Body Temperature,Symptoms (e.g., "painless lumps," "loss of appetite," "depression").

Visualizes disease distribution in livestock populations.
Handles unseen disease labels gracefully.

Dataset & Preprocessing:
Dataset: Contains livestock health records with symptoms and disease labels.

Preprocessing Steps:

1)Label Encoding (sklearn.preprocessing.LabelEncoder).
2)Feature Scaling (StandardScaler).
3)Train-Test Split (80-20).
