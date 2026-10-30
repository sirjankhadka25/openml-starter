import joblib
from sklearn.datasets import load_wine

# Load the saved trained pipeline
model = joblib.load("models/wine_model.joblib")

# Load the wine dataset
wine = load_wine()

# Select one sample
sample = wine.data[0].reshape(1, -1)

# Make prediction
prediction = model.predict(sample)

# Display results
print("Predicted class:", prediction[0])
print("Actual class:", wine.target[0])
print("Predicted class name:", wine.target_names[prediction[0]])