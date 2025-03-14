import streamlit as st
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Set the title of the app
st.title("🍷 Wine Quality Prediction App")

# Define paths (same directory as app.py)
model_path = 'best_wine_quality_model.joblib'
scaler_path = 'scaler.joblib'
data_path = 'winequality.csv'

# Load model and scaler with compatibility fix
try:
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)
except Exception as e:
    st.error(f"Error loading model/scaler: {str(e)}")
    st.stop()

# Load dataset
try:
    df = pd.read_csv(data_path)
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Feature alignment
required_features = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]

# Sidebar inputs
st.sidebar.header("⚙️ Input Features")
input_features = {}
for feature in required_features:
    input_features[feature] = st.sidebar.number_input(
        feature, 
        value=0.0,
        min_value=0.0,
        format="%.2f"
    )

# Prediction
try:
    input_df = pd.DataFrame([input_features])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0] + 3  # Assuming quality scale 3-8
    st.subheader("📊 Prediction Result")
    st.metric(label="Predicted Quality", value=f"{prediction}/10")
except Exception as e:
    st.error(f"Prediction failed: {str(e)}")
    st.stop()

# Visualization section
st.subheader("📈 Model Performance")

# Confusion Matrix
y = df['quality'] - 3
X_scaled = scaler.transform(df[required_features])
y_pred = model.predict(X_scaled)

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y, y_pred), 
            annot=True, fmt="d", 
            cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Classification Report
st.write("#### Classification Metrics")
st.code(classification_report(y, y_pred))

# Accuracy
accuracy = accuracy_score(y, y_pred)
st.write(f"#### Overall Accuracy: {accuracy:.2%}")
