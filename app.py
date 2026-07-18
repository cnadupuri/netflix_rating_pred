%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Netflix Rating Prediction")

type_input = st.selectbox("Type", ["Movie", "TV Show"])
country_input = st.text_input("Country", "India")
release_year = st.number_input("Release Year", 1900, 2035, 2020)
duration = st.text_input("Duration", "90 min")
listed_in = st.text_input("Listed In", "Dramas")

if st.button("Predict"):
    data = pd.DataFrame({
        "type": [type_input],
        "country": [country_input],
        "release_year": [release_year],
        "duration": [duration],
        "listed_in": [listed_in]
    })

    data = preprocessor.transform(data)
    prediction = model.predict(data)

    st.success(f"Predicted Rating: {prediction[0]}")
