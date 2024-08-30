# model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class HeartDiseaseModel:
    def __init__(self):
        self.model = None
        self._load_and_train_model()

    def _load_and_train_model(self):
        # Load and prepare the data
        dataset = pd.read_csv("heart_synthetic_ctgan_with_target.csv")
        X = dataset.drop("target", axis=1)
        y = dataset["target"]

        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        # Train the Random Forest model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, Y_train)

    def predict(self, input_data):
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        probability = self.model.predict_proba(input_data)[0][1] * 100  # Convert to percentage
        return prediction, round(probability, 2)

# Instantiate the model
heart_disease_model = HeartDiseaseModel()
