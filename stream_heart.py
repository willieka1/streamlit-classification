import streamlit as st
import pickle
import pandas as pd

# Fungsi untuk mengambil input dari pengguna
def user_input():
    Age = st.number_input('Umur', min_value=0, max_value=120, step=1)
    Sex = st.selectbox('Jenis Kelamin', ['M', 'F'])
    ChestPainType = st.selectbox('Jenis Nyeri Dada', ['ATA', 'NAP', 'ASY', 'TA'])
    RestingBP = st.number_input('Tekanan Darah Saat Istirahat', min_value=0, step=1)
    Cholesterol = st.number_input('Kolesterol', min_value=0, step=1)
    FastingBS = st.selectbox('Kadar Gula Darah', [0, 1])
    RestingECG = st.selectbox('Hasil Elektrokardiografi', ['Normal', 'ST', 'LVH'])
    MaxHR = st.number_input('Denyut Jantung Maksimum', min_value=0, step=0)
    ExerciseAngina = st.selectbox('Nyeri Dada Saat Aktivitas', ['N', 'Y'])
    Oldpeak = st.number_input('ST Depression', step=0.1)
    ST_Slope = st.selectbox('Slope', ['Up', 'Flat', 'Down'])

    # Mengembalikan input sebagai DataFrame untuk prediksi
    data = {
        'Age': Age,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }
    return pd.DataFrame(data, index=[0])

# Muat model dan label encoders dari file
def load_model_and_encoders():
    with open('heart_failure.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoders.pkl', 'rb') as le_file:
        label_encoders = pickle.load(le_file)
    return model, label_encoders

# Aplikasi Streamlit
def main():
    st.title('Prediksi Penyakit Jantung')
    
    # Ambil input dari pengguna
    user_data = user_input()
    
    # Muat model dan label encoders
    model, label_encoders = load_model_and_encoders()
    
    # Encode fitur kategori
    for column in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        user_data[column] = label_encoders[column].transform(user_data[column])
    
    # Tombol prediksi
    if st.button('Prediksi'):
        # Lakukan prediksi
        prediction = model.predict(user_data)
        
        # Tampilkan hasil prediksi
        if prediction[0] == 0:
            st.write('Kemungkinan Sakit Jantung')
        else:
            st.write('Kemungkinan Tidak Sakit Jantung')

if __name__ == '__main__':
    main()
