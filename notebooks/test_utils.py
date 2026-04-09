import pandas as pd
from utils import automated_pipeline

# 1. Load data mentah
df_raw = pd.read_excel(r'E:\airta drafts\PREDIKSI KADAR HB\data\raw\erm_hd.xlsx')

# 2. Jalankan pipeline
df_clean = automated_pipeline(df_raw)

# 3. Cek Parameter Keberhasilan:
print(f"Jumlah baris awal: {len(df_raw)}")
print(f"Jumlah baris setelah cleaning: {len(df_clean)}")
print(f"Kolom tersedia: {df_clean.columns.tolist()}")

# Cek apakah ada Missing Values yang lolos
print(f"Total Missing Values: {df_clean.isnull().sum().sum()}")

# Cek apakah ID dengan data < 3 sudah hilang
counts = df_clean['id_pasien'].value_counts()
print(f"Pasien dengan data paling sedikit: {counts.min()} bulan")