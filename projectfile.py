import pandas as pd # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

# Step 1: Load the CSV file
# Make sure your CSV has headers: Animal,Age,Temperature,Symptom 1,Symptom 2,Symptom 3,Disease
data = pd.read_csv("D:/Miniproject/animal_disease_dataset.csv")

# Step 2: Encode categorical features
label_encoders = {}
for col in ['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3', 'Disease']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # store encoders for later use

# Step 3: Split into features and target
X = data[['Animal', 'Age', 'Temperature', 'Symptom 1', 'Symptom 2', 'Symptom 3']]
y = data['Disease']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Step 7: Function to make predictions from new data
def predict_disease(animal, age, temperature, symptom1, symptom2, symptom3):
    try:
        input_data = pd.DataFrame([{
            'Animal': label_encoders['Animal'].transform([animal])[0],
            'Age': age,
            'Temperature': temperature,
            'Symptom 1': label_encoders['Symptom 1'].transform([symptom1])[0],
            'Symptom 2': label_encoders['Symptom 2'].transform([symptom2])[0],
            'Symptom 3': label_encoders['Symptom 3'].transform([symptom3])[0]
        }])
        prediction = model.predict(input_data)
        disease = label_encoders['Disease'].inverse_transform(prediction)[0]
        return disease
    except Exception as e:
        return f"Error in prediction: {e}"

# Example usage
example_prediction = predict_disease("cow", 5, 102.5, "depression", "painless lumps", "loss of appetite")
print("Predicted Disease:", example_prediction)
