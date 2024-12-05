import streamlit as st
from data_preprocessing import preprocess_data
from model import train_model
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
data_path = 'src/data.csv'
data = preprocess_data(data_path)

# Sidebar Menu
menu = st.sidebar.selectbox(
    "Navigasi",
    ["Deskripsi", "Dataset", "Visualisasi", "Prediksi Harga", "Tentang Algoritma"]
)

# **Deskripsi**
if menu == "Deskripsi":
    st.title("Deskripsi Aplikasi")
    st.markdown("""
    Aplikasi ini dirancang untuk memprediksi harga rumah berdasarkan beberapa fitur seperti jumlah kamar tidur, kamar mandi, luas bangunan, dan luas tanah.  
    Model yang digunakan adalah **Linear Regression** untuk menghasilkan prediksi harga yang akurat.  
    """)

# **Dataset**
elif menu == "Dataset":
    st.title("Dataset")
    st.markdown("""
    Dataset ini memuat informasi harga rumah, jumlah kamar tidur, jumlah kamar mandi, luas bangunan, luas tanah, dan beberapa fitur lainnya.  
    """)
    if st.checkbox("Tampilkan Data (5 Baris Awal)"):
        st.write(data.head())

# **Visualisasi Data**
elif menu == "Visualisasi":
    st.title("Visualisasi Data")
    st.markdown("""
    Anda dapat memvisualisasikan data dalam bentuk histogram atau scatter plot.
    """)

    if st.checkbox("Tampilkan Histogram"):
        col_to_plot = st.selectbox("Pilih Kolom untuk Histogram", data.columns)
        fig, ax = plt.subplots()
        ax.hist(data[col_to_plot], bins=30, color='skyblue', edgecolor='black')
        st.pyplot(fig)

    if st.checkbox("Tampilkan Scatter Plot"):
        x_col = st.selectbox("Pilih Kolom X", data.columns)
        y_col = st.selectbox("Pilih Kolom Y", data.columns)
        fig, ax = plt.subplots()
        ax.scatter(data[x_col], data[y_col], alpha=0.6, color='orange')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)

# **Prediksi Harga**
elif menu == "Prediksi Harga":
    st.title("Prediksi Harga Rumah")
    st.markdown("""
    Masukkan fitur rumah di bawah ini untuk mendapatkan prediksi harga:
    """)

    # Input fitur rumah
    bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1)
    bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1)
    sqft_living = st.number_input("Luas Bangunan (sqft)", min_value=1)
    sqft_lot = st.number_input("Luas Tanah (sqft)", min_value=1)

    # Muat model dan daftar fitur
    model = train_model(data)  # Training model dan menyimpan fitur
    feature_names = joblib.load('features.pkl')  # Muat fitur yang disimpan

    if st.button("Prediksi Harga"):
        # Input fitur
        features = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot]], 
                                columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot'])

        # Tambahkan fitur yang hilang
        for feature in feature_names:
            if feature not in features.columns:
                features[feature] = 0  # Nilai default

        # Prediksi harga
        price = model.predict(features)

        # Tampilkan hanya harga prediksi
        st.write(f"Harga yang diprediksi: ${price[0]:,.2f}")

# **Tentang Algoritma**
elif menu == "Tentang Algoritma":
    st.title("Tentang Algoritma")
    st.markdown("""
    ### Linear Regression
    Algoritma **Linear Regression** digunakan untuk memodelkan hubungan antara fitur (independen) dan target (dependen) dengan persamaan linear.
    Model ini cocok untuk data yang memiliki hubungan linear dan menghasilkan prediksi nilai kontinu.

    #### Proses Pelatihan:
    1. Membagi data menjadi **training set** (80%) dan **testing set** (20%).
    2. Melatih model menggunakan data training.
    3. Mengevaluasi model dengan metrik **Mean Squared Error (MSE)**.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Pembuat**: Rifki Muhammad Ilyasa")
