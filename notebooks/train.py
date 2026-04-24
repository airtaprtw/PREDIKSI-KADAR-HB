import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMRegressor
from utils import automated_pipeline

# Path tetap untuk menyimpan master data mentah
MASTER_DATA_PATH = 'data/master_data_mentah.xlsx'

def run_retraining(file_path):
    """
    Pipeline otomatis: Data Mentah -> Merge dengan Master -> 
    Cleaning -> Feature Engineering -> Training -> Save Model
    
    Sistem akan:
    1. Membaca data baru yang diupload
    2. Merge dengan master data lama (jika ada)
    3. Deduplikasi baris yang persis sama
    4. Simpan hasil merge sebagai master data terbaru
    5. Jalankan training ulang dengan data gabungan
    """
    try:
        # 1. LOAD DATA BARU YANG DIUPLOAD
        print(f"[*] Memuat data baru dari: {file_path}")
        df_baru = pd.read_excel(file_path)
        print(f"    -> {len(df_baru)} baris data baru ditemukan")

        # 2. MERGE DENGAN MASTER DATA LAMA (jika ada)
        os.makedirs('data', exist_ok=True)

        if os.path.exists(MASTER_DATA_PATH):
            print(f"[*] Ditemukan master data lama. Melakukan merge...")
            df_lama = pd.read_excel(MASTER_DATA_PATH)
            print(f"    -> {len(df_lama)} baris data lama ditemukan")

            # Gabungkan data lama + data baru
            df_raw = pd.concat([df_lama, df_baru], ignore_index=True)
            print(f"    -> Total setelah concat: {len(df_raw)} baris")

            # Deduplikasi: hapus baris yang identik persis
            # (kasus: pasien yang sama & tanggal sama muncul di data lama dan baru)
            df_raw = df_raw.drop_duplicates().reset_index(drop=True)
            print(f"    -> Total setelah deduplikasi: {len(df_raw)} baris")
        else:
            print(f"[*] Belum ada master data. Menggunakan data baru sebagai data awal.")
            df_raw = df_baru

        # 3. SIMPAN HASIL MERGE SEBAGAI MASTER DATA TERBARU
        df_raw.to_excel(MASTER_DATA_PATH, index=False)
        print(f"[*] Master data berhasil diperbarui di: {MASTER_DATA_PATH}")
        
        # 4. AUTOMATED CLEANING (Memanggil fungsi dari utils.py)
        print("[*] Menjalankan automated cleaning via utils.py...")
        df = automated_pipeline(df_raw)
        
        # 5. FEATURE ENGINEERING (Sesuai Logika Riset)
        print("[*] Membangun fitur lag dan indikator klinis...")
        # Pastikan data urut per pasien & waktu
        df = df.sort_values(by=['id_pasien', 'tgl_pemeriksaan']).reset_index(drop=True)
        
        # Lag Features
        df['hb_lag'] = df.groupby('id_pasien')['hemoglobin'].shift(1)
        df['hb_lag2'] = df.groupby('id_pasien')['hemoglobin'].shift(2)
        df['hb_delta'] = df['hb_lag'] - df['hb_lag2']
        
        # Klinis Indikator (Inflamasi & EPO Resistance)
        df['inflamasi'] = (df['leukosit'] / 10000) * (df['trombosit'] / 150000)
        df['epo_resist'] = df['epo'] / (df['inflamasi'] + 1)
        
        # Drop baris yang kosong akibat shift()
        df = df.dropna(subset=['hb_lag', 'hb_lag2']).reset_index(drop=True)
        
        # 6. SELEKSI FITUR (Sesuai X yang kamu gunakan di Notebook)
        # Menghapus kolom yang tidak relevan untuk modeling
        cols_to_drop = [
            'id_pasien', 'tgl_pemeriksaan', 'hemoglobin', 
            'hematokrit', 'eritrosit', 'MCH', 'epo', 'inflamasi'
        ]
        
        X = df.drop(columns=cols_to_drop)
        y = df['hemoglobin']
        
        print(f"[*] Fitur yang digunakan: {X.columns.tolist()}")
        
        # 7. TRAINING MODEL (Hanya LightGBM - Algoritma Terpilih)
        print("[*] Melatih ulang model LightGBM dengan Best Parameters...")
        
        # Gunakan best_params hasil Grid Search kamu
        best_params = {
            'n_estimators': 100,      # Sesuaikan dengan hasil grid_lgbm.best_params_
            'max_depth': 5,           # Sesuaikan dengan hasil grid_lgbm.best_params_
            'learning_rate': 0.05,    # Sesuaikan dengan hasil grid_lgbm.best_params_
            'num_leaves': 15,         # Sesuaikan dengan hasil grid_lgbm.best_params_
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMRegressor(**best_params)
        # Tambahkan ini di train.py sebelum model.fit(X, y)
        features_order = ['usia', 'jk', 'MCHC', 'MCV', 'leukosit', 'trombosit', 'hb_lag', 'hb_delta', 'epo_resist']
        X = X[features_order]
        model.fit(X, y)
        
        # 8. PENYIMPANAN MODEL (.pkl)
        # Membuat folder models jika belum ada
        if not os.path.exists('models'):
            os.makedirs('models')
            
        model_save_path = 'models/lgbm_best_model.pkl'
        joblib.dump(model, model_save_path)
        
        print(f"[SUCCESS] Model berhasil diperbarui dan disimpan di: {model_save_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Terjadi kegagalan pada pipeline: {e}")
        return False

if __name__ == "__main__":
    # Tentukan path file mentah secara default untuk pengujian lokal
    DEFAULT_FILE = r'E:\airta drafts\PREDIKSI KADAR HB\data\raw\erm_hd.xlsx'
    run_retraining(DEFAULT_FILE)