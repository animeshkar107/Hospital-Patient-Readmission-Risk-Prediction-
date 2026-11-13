
import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
saved_data = joblib.load('readmission_model.pkl')

model = saved_data['model']
preprocessor = saved_data['preprocessor']

st.set_page_config(
    page_title="Hospital Readmission Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("Hospital Patient Readmission Risk Prediction")

st.write(
    "Upload a CSV file containing patient encounter features "
    "to predict readmission within 30 days."
)

uploaded_file = st.file_uploader(
    "Upload Patient CSV",
    type=['csv']
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Remove identifiers and target columns if present
    data = data.drop(
        columns=[
            'encounter_id',
            'patient_nbr',
            'readmitted',
            'readmission_risk'
        ],
        errors='ignore'
    )

    try:
        processed_data = preprocessor.transform(data)
        predictions = model.predict(processed_data)

        results = data.copy()

        results['Predicted_Readmission'] = [
            'Within 30 Days' if value == 1
            else 'Not Within 30 Days'
            for value in predictions
        ]

        st.subheader("Prediction Results")
        st.dataframe(results)

        st.download_button(
            "Download Predictions",
            results.to_csv(index=False),
            "readmission_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Please upload a CSV containing all required "
            "model input columns."
        )

st.warning(
    "Educational project only. Predictions are not intended "
    "for clinical decisions or patient care."
)
