import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🩺 Breast Cancer Prediction App")
st.write("Enter tumor feature values to predict cancer type")

st.sidebar.header("Input Features")

# User input
def user_input_features():
    inputs = {}
    for feature in data.feature_names:
        inputs[feature] = st.sidebar.slider(
            feature,
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].mean())
        )
    return pd.DataFrame([inputs])

input_df = user_input_features()

st.subheader("User Input Values")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Output
st.subheader("Prediction Result")

if prediction[0] == 1:
    st.success("🟢 Benign (Non-cancerous)")
else:
    st.error("🔴 Malignant (Cancerous)")

st.subheader("Prediction Probability")
st.write(pd.DataFrame(
    prediction_proba,
    columns=["Malignant", "Benign"]
))
