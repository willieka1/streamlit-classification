import streamlit as st
import pickle
import pandas as pd

# Fungsi untuk mengambil input dari pengguna
def user_input():
    uranium_lead_ratio = st.number_input('Uranium-Lead Ratio', step=0.01)
    carbon_14_ratio = st.number_input('Carbon-14 Ratio', step=0.01)
    radioactive_decay_series = st.number_input('Radioactive Decay Series', step=0.01)
    stratigraphic_layer_depth = st.number_input('Stratigraphic Layer Depth (meters)', min_value=0.0, step=0.1)
    geological_period = st.selectbox('Geological Period', ['Cambrian', 'Ordovician', 'Silurian', 'Devonian', 'Carboniferous', 'Permian', 'Triassic', 'Jurassic', 'Cretaceous', 'Paleogene', 'Neogene', 'Quaternary'])
    paleomagnetic_data = st.selectbox('Paleomagnetic Data', ['Normal polarity', 'Reversed polarity'])
    inclusion_of_other_fossils = st.selectbox('Inclusion of Other Fossils', ['True', 'False'])
    isotopic_composition = st.number_input('Isotopic Composition', step=0.01)
    surrounding_rock_type = st.selectbox('Surrounding Rock Type', ['Conglomerate', 'Sandstone', 'Shale', 'Limestone', 'Others'])
    stratigraphic_position = st.selectbox('Stratigraphic Position', ['Low', 'Middle', 'High'])
    fossil_size = st.number_input('Fossil Size (cm)', step=0.1)
    fossil_weight = st.number_input('Fossil Weight (g)', step=0.1)

    # Mengembalikan input sebagai DataFrame untuk prediksi
    data = {
        'uranium_lead_ratio': uranium_lead_ratio,
        'carbon_14_ratio': carbon_14_ratio,
        'radioactive_decay_series': radioactive_decay_series,
        'stratigraphic_layer_depth': stratigraphic_layer_depth,
        'geological_period': geological_period,
        'paleomagnetic_data': paleomagnetic_data,
        'inclusion_of_other_fossils': inclusion_of_other_fossils,
        'isotopic_composition': isotopic_composition,
        'surrounding_rock_type': surrounding_rock_type,
        'stratigraphic_position': stratigraphic_position,
        'fossil_size': fossil_size,
        'fossil_weight': fossil_weight
    }
    return pd.DataFrame(data, index=[0])

# Muat model dan label encoders dari file
def load_model_and_encoders():
    with open('Age_Fossil.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoders_age.pkl', 'rb') as le_file:
        label_encoders = pickle.load(le_file)
    return model, label_encoders

# Aplikasi Streamlit
def main():
    st.title('Analisis Regresi Geologi dan Paleontologi')
    
    # Ambil input dari pengguna
    user_data = user_input()
    
    # Muat model dan label encoders
    model, label_encoders = load_model_and_encoders()
    
    # Encode fitur kategori
    categorical_columns = {
        'geological_period': 'Geological Period',
        'paleomagnetic_data': 'Paleomagnetic Data',
        'inclusion_of_other_fossils': 'Inclusion of Other Fossils',
        'surrounding_rock_type': 'Surrounding Rock Type',
        'stratigraphic_position': 'Stratigraphic Position'
    }
    
    for column, label in categorical_columns.items():
        if label in label_encoders:
            user_data[column] = label_encoders[label].transform(user_data[column])
    
    # Tombol prediksi
    if st.button('Prediksi'):
        # Lakukan prediksi
        prediction = model.predict(user_data)
        
        # Tampilkan hasil prediksi
        st.write(f'Predicted Age: {prediction[0]:,.2f} years')

if __name__ == '__main__':
    main()
