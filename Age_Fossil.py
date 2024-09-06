import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Menyiapkan aplikasi Streamlit
st.title("Prediksi Usia Fosil dengan Regresi Linier, Random Forest, dan XGBoost")

# Unggah dataset
uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
if uploaded_file is not None:
    # Muat data
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.write(data.head())

    # Mendefinisikan fitur dan target
    feature_columns = [
        'uranium_lead_ratio', 'carbon_14_ratio', 'radioactive_decay_series',
        'stratigraphic_layer_depth', 'geological_period', 'paleomagnetic_data',
        'inclusion_of_other_fossils', 'isotopic_composition', 'surrounding_rock_type',
        'stratigraphic_position', 'fossil_size', 'fossil_weight'
    ]
    target_column = 'age'  # Mengasumsikan 'age' adalah kolom dengan usia fosil

    # Periksa apakah kolom target ada
    if target_column not in data.columns:
        st.error(f"Kolom '{target_column}' tidak ditemukan di dataset.")
    else:
        # Tampilkan opsi fitur
        st.sidebar.header("Pilih Kolom")
        features = st.sidebar.multiselect("Pilih kolom fitur", options=feature_columns, default=feature_columns)
        
        # Pilih model
        model_option = st.sidebar.selectbox(
            "Pilih model",
            ["Regresi Linier", "Random Forest", "XGBoost"]
        )
        
        if len(features) > 0:
            # Periksa apakah fitur yang dipilih ada dalam dataset
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                st.error(f"Kolom fitur berikut tidak ditemukan di dataset: {', '.join(missing_features)}")
            else:
                # Tangani data kategorikal
                data = pd.get_dummies(data, columns=[col for col in features if data[col].dtype == 'object'], drop_first=True)
                
                # Periksa apakah semua fitur yang dipilih ada setelah pengkodean
                features = [col for col in features if col in data.columns]

                # Siapkan dataset fitur dan target
                X = data[features]
                y = data[target_column]

                # Periksa nilai yang hilang
                if X.isnull().any().any() or y.isnull().any():
                    st.error("Data mengandung nilai yang hilang. Harap bersihkan data Anda.")
                else:
                    # Pisahkan data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Pilih model
                    if model_option == "Regresi Linier":
                        model = LinearRegression()
                    elif model_option == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_option == "XGBoost":
                        model = XGBRegressor(n_estimators=100, random_state=42)
                    
                    # Latih model
                    model.fit(X_train, y_train)

                    # Prediksi
                    y_pred = model.predict(X_test)

                    # Metode evaluasi
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write("Mean Squared Error:", mse)
                    st.write("R-squared:", r2)

                    # Plot
                    st.subheader("Visualisasi")
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
                    ax.set_xlabel('Nilai Aktual')
                    ax.set_ylabel('Nilai Prediksi')
                    ax.set_title(f'Nilai Aktual vs Prediksi ({model_option})')
                    st.pyplot(fig)

                    # Tampilkan koefisien atau fitur penting
                    if model_option == "Regresi Linier":
                        st.write("Koefisien regresi:")
                        for feature, coef in zip(features, model.coef_):
                            st.write(f"{feature}: {coef}")

                        st.write("Intercept:", model.intercept_)
                    elif model_option in ["Random Forest", "XGBoost"]:
                        st.write("Fitur penting:")
                        feature_importances = model.feature_importances_
                        for feature, importance in zip(features, feature_importances):
                            st.write(f"{feature}: {importance}")
        else:
            st.warning("Silakan pilih kolom fitur untuk model.")
else:
    st.info("Silakan unggah file CSV untuk memulai.")
