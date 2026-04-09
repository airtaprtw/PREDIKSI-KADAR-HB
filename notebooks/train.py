import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMRegressor
from utils import automated_pipeline

def run_retraining(file_path):
    """
    Pipeline otomatis: Data Mentah -> Cleaning -> Feature Engineering -> Training -> Save Model
    """
    try:
        # 1. LOAD DATA MENTAH
        print(f"[*] Memuat data mentah dari: {file_path}")
        df_raw = pd.read_excel(file_path)
        
        # 2. AUTOMATED CLEANING (Memanggil fungsi dari utils.py)
        print("[*] Menjalankan automated cleaning via utils.py...")
        df = automated_pipeline(df_raw)
        
        # 3. FEATURE ENGINEERING (Sesuai Logika Riset)
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
        
        # 4. SELEKSI FITUR (Sesuai X yang kamu gunakan di Notebook)
        # Menghapus kolom yang tidak relevan untuk modeling
        cols_to_drop = [
            'id_pasien', 'tgl_pemeriksaan', 'hemoglobin', 
            'hematokrit', 'eritrosit', 'MCH', 'epo', 'inflamasi'
        ]
        
        X = df.drop(columns=cols_to_drop)
        y = df['hemoglobin']
        
        print(f"[*] Fitur yang digunakan: {X.columns.tolist()}")
        
        # 5. TRAINING MODEL (Hanya LightGBM - Algoritma Terpilih)
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
        
        # 6. PENYIMPANAN MODEL (.pkl)
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