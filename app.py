import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv("CC GENERAL.csv")

# Streamlit title and data preview
st.title('Kartu Kredit')
st.subheader('Data Kartu Kredit')
st.write(df.head())

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_numeric = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns

# Impute missing values
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

# Visualize heatmap
st.subheader('Visualisasi Data Kartu Kredit')
st.write('Visualisasi Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_numeric_imputed.corr(), annot=True, cmap='coolwarm', ax=ax)  # Use df_numeric_imputed for corr
st.pyplot(fig)

# Visualize distributions of numeric features
st.write('Distribusi Fitur Numerik')
fig, ax = plt.subplots(figsize=(15, 10))
df_numeric_imputed.hist(bins=30, ax=ax)  # Use df_numeric_imputed for hist
plt.suptitle('Distribusi Fitur Numerik')
st.pyplot(fig)

# Cluster visualization
st.subheader('Visualisasi Cluster')
n_cluster = st.slider("Pilih jumlah cluster:", min_value=2, max_value=10, value=3)  # Increased max_value for flexibility

# Standardize features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric_imputed), columns=df_numeric_imputed.columns)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_cluster, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualize clusters
st.write('Visualisasi Cluster')
fig, ax = plt.subplots()
# Ensure 'CREDIT_LIMIT' and 'BALANCE' exist in df_numeric_imputed
if 'CREDIT_LIMIT' in df.columns and 'BALANCE' in df.columns:
    sns.scatterplot(data=df, x='CREDIT_LIMIT', y='BALANCE', hue='Cluster', palette='Set1', ax=ax)
    plt.title('Visualisasi Data Kartu Kredit dengan Clustering')
else:
    st.write("Kolom 'CREDIT_LIMIT' atau 'BALANCE' tidak ditemukan dalam data.")

st.pyplot(fig)

# Display count of data points in each cluster
st.write('Jumlah Data Setiap Cluster')
st.write(df['Cluster'].value_counts())
