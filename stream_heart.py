import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the saved model
with open("heart_failure.sav", "rb") as f:
    model = pickle.load(f)

# Web page title
st.title('Prediksi Penyakit Jantung')

# User input function
def user_input():
    Age = st.number_input('Umur', min_value=0, max_value=120, step=1)
    Sex = st.selectbox('Jenis Kelamin', [0,1])
    ChestPainType = st.selectbox('Jenis Nyeri Dada', [0,1,2,3])
    RestingBP = st.number_input('Tekanan Darah Saat Istirahat', min_value=0, step=1)
    Cholesterol = st.number_input('Kolesterol', min_value=0, step=1)
    FastingBS = st.selectbox('Kadar Gula Darah', [0, 1])  
    RestingECG = st.selectbox('Hasil Elektrokardiografi', [0,1,2])
    MaxHR = st.number_input('Denyut Jantung Maksimum', min_value=0, step=1)
    ExerciseAngina = st.selectbox('Nyeri Dada Saat Aktivitas', [0,1])
    Oldpeak = st.number_input('ST Depression', step=0.1)
    ST_Slope = st.selectbox('Slope', [0, 1, 2])

    data = {'Age': Age, 'Sex': Sex, 'ChestPainType': ChestPainType,
            'RestingBP': RestingBP, 'Cholesterol': Cholesterol,
            'FastingBS': FastingBS, 'RestingECG': RestingECG,
            'MaxHR': MaxHR, 'ExerciseAngina': ExerciseAngina,
            'Oldpeak': Oldpeak, 'ST_Slope': ST_Slope}

    df = pd.DataFrame(data, index=[0])
    return df

input_df = user_input()

# Predict
if st.button('Prediksi Penyakit Jantung'):
    # Prepare the input for the model
    input_data = input_df.values
    
    # Convert categorical features to numeric if necessary
    # Example: encode categorical variables here if your model needs it
    
    # Make prediction
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        heart_diagnosis = 'Anda mungkin mengalami penyakit jantung'
    else:
        heart_diagnosis = 'Anda tidak mungkin mengalami penyakit jantung'

    st.subheader(heart_diagnosis)
