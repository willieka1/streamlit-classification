import streamlit as st
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Load the model pipeline
with open('Age_Fossil.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Load label encoders
with open('label_encoders_age.pkl', 'rb') as le_file:
    label_encoders_age = pickle.load(le_file)

# Function to encode categorical variables
def encode_categorical_feature(value, encoder):
    if encoder:
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            # Handle unseen categories
            return -1  # or some default value appropriate for your model
    else:
        # If encoder is not available, return a default value
        return -1

# Streamlit app
st.title('Fossil Age Prediction')

# User input
uranium_lead_ratio = st.number_input('Uranium-Lead Ratio', step=0.01)
carbon_14_ratio = st.number_input('Carbon-14 Ratio', step=0.01)
radioactive_decay_series = st.number_input('Radioactive Decay Series', step=0.01)
stratigraphic_layer_depth = st.number_input('Stratigraphic Layer Depth (meters)', min_value=0.0, step=0.1)

# Encode categorical features
geological_period = st.selectbox('Geological Period', ['Cambrian', 'Ordovician', 'Silurian', 'Devonian', 'Carboniferous', 'Permian', 'Triassic', 'Jurassic', 'Cretaceous', 'Paleogene', 'Neogene', 'Quaternary'])
paleomagnetic_data = st.selectbox('Paleomagnetic Data', ['Normal polarity', 'Reversed polarity'])
inclusion_of_other_fossils = st.selectbox('Inclusion of Other Fossils', ['True', 'False'])
isotopic_composition = st.number_input('Isotopic Composition', step=0.01)
surrounding_rock_type = st.selectbox('Surrounding Rock Type', ['Conglomerate', 'Sandstone', 'Shale', 'Limestone', 'Others'])
stratigraphic_position = st.selectbox('Stratigraphic Position', ['Bottom', 'Middle', 'Top'])
fossil_size = st.number_input('Fossil Size (cm)', step=0.1)
fossil_weight = st.number_input('Fossil Weight (g)', step=0.1)

# Prepare input data
data = [
    uranium_lead_ratio,
    carbon_14_ratio,
    radioactive_decay_series,
    stratigraphic_layer_depth,
    encode_categorical_feature(geological_period, label_encoders_age.get('geological_period')),
    encode_categorical_feature(paleomagnetic_data, label_encoders_age.get('paleomagnetic_data')),
    encode_categorical_feature(inclusion_of_other_fossils, label_encoders_age.get('inclusion_of_other_fossils')),
    isotopic_composition,
    encode_categorical_feature(surrounding_rock_type, label_encoders_age.get('surrounding_rock_type')),
    encode_categorical_feature(stratigraphic_position, label_encoders_age.get('stratigraphic_position')),
    fossil_size,
    fossil_weight
]

# Convert data to numpy array and reshape for prediction
input_data = np.array(data).reshape(1, -1)

# Predict using the model pipeline
if st.button('Predict Age'):
    try:
        prediction = pipeline.predict(input_data)
        st.write('**Predicted Age of Fossil:**')
        st.write(prediction[0])
    except Exception as e:
        st.write('Error in prediction:', str(e))
