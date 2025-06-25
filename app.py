import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sidebar - Identitas
st.sidebar.markdown('<div style="display:flex;justify-content:center;"><img src="https://avatars.githubusercontent.com/u/9919?s=200&v=4" width="120" style="border-radius:50%;margin-bottom:16px;"/></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="text-align:center;"><h3>Humam Wajdi Manaf</h3><p><strong>NIM:</strong> A11.2023.15428</p></div>', unsafe_allow_html=True)
st.sidebar.markdown('''
# Prediksi Harga Rumah
---

### Deskripsi Projek
Aplikasi ini memprediksi harga rumah berdasarkan fitur:
- Luas bangunan
- Luas tanah
- Jumlah kamar tidur
- Kamar mandi
- Garasi

Model yang digunakan: **Regresi Linier**
''')

st.markdown('<h1 style="text-align:center; color:#4F8BF9;">Prediksi Harga Rumah</h1>', unsafe_allow_html=True)
st.markdown('<hr style="height:5px;border:none;color:#4F8BF9;background-color:#4F8BF9;">', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Masukkan fitur rumah di bawah ini untuk mendapatkan estimasi harga rumah secara otomatis.</p>', unsafe_allow_html=True)

# Load data
data_file = 'DATA RUMAH.xlsx'
if os.path.exists(data_file):
    df = pd.read_excel(data_file)
    st.sidebar.success('Data berhasil dimuat!')
else:
    st.error(f'File {data_file} tidak ditemukan di folder. Pastikan file sudah ada.')
    st.stop()

# Preprocessing
df.drop(['NO', 'NAMA RUMAH'], axis=1, inplace=True, errors='ignore')
df.rename(columns={
    'LB': 'luas_bangunan',
    'LT': 'luas_tanah',
    'KT': 'kamar_tidur',
    'KM': 'kamar_mandi',
    'GRS': 'garasi',
    'HARGA': 'harga_rumah'
}, inplace=True)

# Modeling
x = df.drop('harga_rumah', axis=1)
y = df['harga_rumah']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi harga rumah
def predict_house_price(luas_bangunan, luas_tanah, kamar_tidur, kamar_mandi, garasi):
    input_data = pd.DataFrame([[luas_bangunan, luas_tanah, kamar_tidur, kamar_mandi, garasi]], columns=x.columns)
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Input Prediksi di Main Page
with st.container():
    st.markdown('<h3 style="text-align:center;">Input Fitur Rumah</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        luas_bangunan = st.number_input('Luas Bangunan (m2)', min_value=50, max_value=500, value=250, step=10, help='Masukkan luas bangunan rumah dalam meter persegi')
        kamar_tidur = st.selectbox('Jumlah Kamar Tidur', options=list(range(1, 11)), index=1, help='Pilih jumlah kamar tidur')
        garasi = st.selectbox('Jumlah Garasi', options=list(range(0, 6)), index=1, help='Pilih jumlah garasi')
    with col2:
        luas_tanah = st.number_input('Luas Tanah (m2)', min_value=100, max_value=1000, value=500, step=10, help='Masukkan luas tanah rumah dalam meter persegi')
        kamar_mandi = st.selectbox('Jumlah Kamar Mandi', options=list(range(1, 6)), index=0, help='Pilih jumlah kamar mandi')

    prediksi = None
    if st.button('Prediksi Harga Rumah', use_container_width=True):
        prediksi = predict_house_price(luas_bangunan, luas_tanah, kamar_tidur, kamar_mandi, garasi)
        st.balloons()

    if prediksi is not None:
        st.markdown(f'''
        <div style="background-color:#fff3e0;padding:20px;border-radius:10px;margin-top:20px;text-align:center;box-shadow:0 2px 8px rgba(244,67,54,0.08);border:2px solid #f44336;">
            <h2 style="color:#f44336;">Estimasi Harga Rumah</h2>
            <h1 style="color:#388e3c;">Rp {prediksi:,.0f}</h1>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Evaluasi Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mse)

# Evaluasi Model
st.markdown('<hr style="height:2px;border:none;color:#e0e0e0;background-color:#e0e0e0;">', unsafe_allow_html=True)
st.markdown('<h2 style="text-align:center;">Evaluasi Model</h2>', unsafe_allow_html=True)

# Tabel atas: MSE dan R2
st.markdown('<div style="text-align:center;"><b>Tabel Evaluasi Utama</b></div>', unsafe_allow_html=True)
table_atas = pd.DataFrame({
    'Metrik': ['Mean Squared Error', 'R-squared'],
    'Nilai': [f'{mse:,.2f}', f'{r2:.2f}']
})
st.markdown('<div style="display:flex;justify-content:center;">', unsafe_allow_html=True)
st.table(table_atas)
st.markdown('</div>', unsafe_allow_html=True)

# Tabel bawah: MAE, RMSE, Akurasi Prediksi
st.markdown('<div style="text-align:center;"><b>Tabel Evaluasi Lanjutan</b></div>', unsafe_allow_html=True)
table_bawah = pd.DataFrame({
    'Metrik': ['Mean Absolute Error', 'Root Mean Squared Error', 'Akurasi Prediksi'],
    'Nilai': [f'{mae:,.2f}', f'{rmse:,.2f}', f'{model.score(X_test, y_test)*100:.2f}%']
})
st.markdown('<div style="display:flex;justify-content:center;">', unsafe_allow_html=True)
st.table(table_bawah)
st.markdown('</div>', unsafe_allow_html=True)

# Visualisasi
st.markdown('<hr style="height:2px;border:none;color:#e0e0e0;background-color:#e0e0e0;">', unsafe_allow_html=True)
st.markdown('<h2 style="text-align:center;">Visualisasi Data dan Model</h2>', unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='#f44336', alpha=0.7)
ax.set_xlabel('Nilai Sebenarnya')
ax.set_ylabel('Nilai Prediksi')
ax.set_title('Nilai Sebenarnya vs Nilai Prediksi')
st.pyplot(fig)
