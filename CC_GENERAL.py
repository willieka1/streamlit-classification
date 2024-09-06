import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Memuat data
df = pd.read_csv("CC GENERAL.csv")

# Judul Streamlit dan pratinjau data
st.title('Kartu Kredit')
st.subheader('Data Kartu Kredit')
st.write(df.head())

# Menangani nilai yang hilang
imputer = SimpleImputer(strategy='mean')
df_numeric = df.select_dtypes(include=['float64', 'int64'])  # Memilih hanya kolom numerik

# Imputasi nilai yang hilang
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

# Visualisasi heatmap
st.subheader('Visualisasi Data Kartu Kredit')
st.write('Visualisasi Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_numeric_imputed.corr(), annot=True, cmap='coolwarm', ax=ax)  # Menggunakan df_numeric_imputed untuk korelasi
st.pyplot(fig)

# Visualisasi distribusi fitur numerik
st.write('Distribusi Fitur Numerik')
fig, ax = plt.subplots(figsize=(15, 10))
df_numeric_imputed.hist(bins=30, ax=ax)  
plt.suptitle('Distribusi Fitur Numerik')
st.pyplot(fig)

# Visualisasi cluster
st.subheader('Visualisasi Cluster')
n_cluster = st.slider("Pilih jumlah cluster:", min_value=2, max_value=10, value=3)  

# Standarisasi fitur
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric_imputed), columns=df_numeric_imputed.columns)

# Melakukan clustering KMeans
kmeans = KMeans(n_clusters=n_cluster, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualisasi cluster
st.write('Visualisasi Cluster')
fig, ax = plt.subplots()
# Memastikan 'CREDIT_LIMIT' dan 'BALANCE' ada di df_numeric_imputed
if 'CREDIT_LIMIT' in df.columns and 'BALANCE' in df.columns:
    sns.scatterplot(data=df, x='CREDIT_LIMIT', y='BALANCE', hue='Cluster', palette='Set1', ax=ax)
    plt.title('Visualisasi Data Kartu Kredit dengan Clustering')
else:
    st.write("Kolom 'CREDIT_LIMIT' atau 'BALANCE' tidak ditemukan dalam data.")

st.pyplot(fig)

# Menampilkan jumlah data di setiap cluster
st.write('Jumlah Data Setiap Cluster')
st.write(df['Cluster'].value_counts())
