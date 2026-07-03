import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import automated_pipeline, register_new_model

def run_retraining(file_path):
    """
    Pipeline retraining: Cleaning -> Feature Engineering -> Training -> Save Model

    PENTING: file_path yang diterima di sini DIASUMSIKAN SUDAH berisi
    seluruh data gabungan terbaru (data lama + data baru, sudah
    dideduplikasi). Proses merge & deduplikasi dilakukan di sisi
    PEMANGGIL (main.py / halaman Retraining System), BUKAN di sini.

    Ini sengaja dibuat begitu supaya hanya ada SATU sumber kebenaran
    untuk master data (dikelola oleh main.py di app/data/master_data_mentah.xlsx).
    Sebelumnya, fungsi ini juga melakukan merge dengan file master-nya
    sendiri (path relatif 'data/master_data_mentah.xlsx'), yang bisa
    mengarah ke lokasi berbeda dari master data milik main.py tergantung
    working directory saat Streamlit dijalankan -- inilah yang menyebabkan
    jumlah baris data / hasil retraining bisa tidak ter-update dengan benar.
    """
    try:
        # 1. LOAD DATA (SUDAH GABUNGAN LENGKAP, DIKIRIM OLEH main.py)
        print(f"[*] Memuat data untuk training dari: {file_path}")
        df_raw = pd.read_excel(file_path)
        print(f"    -> Total {len(df_raw)} baris data (termasuk seluruh histori yang sudah digabung)")

        # 4. AUTOMATED CLEANING (Memanggil fungsi dari utils.py)
        print("[*] Menjalankan automated cleaning via utils.py...")
        df = automated_pipeline(df_raw)
        print(f"    -> {len(df)} baris setelah cleaning & agregasi bulanan")
        
        # 5. FEATURE ENGINEERING 
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
        print(f"    -> {len(df)} baris siap dipakai untuk training (setelah drop baris awal per pasien akibat lag)")
        
        # 6. SELEKSI FITUR 
        cols_to_drop = [
            'id_pasien', 'tgl_pemeriksaan', 'hemoglobin', 
            'hematokrit', 'eritrosit', 'MCH', 'epo', 'inflamasi'
        ]
        
        X = df.drop(columns=cols_to_drop)
        y = df['hemoglobin']
        
        print(f"[*] Fitur yang digunakan: {X.columns.tolist()}")
        
        # 7. TRAINING MODEL (Hanya LightGBM - Algoritma Terpilih)
        print("[*] Melatih ulang model LightGBM dengan Best Parameters...")
        
        # Gunakan best_params hasil Grid Search
        best_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.05,
            'num_leaves': 15,
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMRegressor(**best_params)
        features_order = ['usia', 'jk', 'MCHC', 'MCV', 'leukosit', 'trombosit', 'hb_lag', 'hb_delta', 'epo_resist']
        X = X[features_order]
        model.fit(X, y)

        # Metrik sederhana pada data latih (bukan validasi terpisah),
        # hanya sebagai catatan referensi di riwayat model.
        y_pred_train = model.predict(X)
        train_rmse = float(np.sqrt(mean_squared_error(y, y_pred_train)))
        train_mae  = float(mean_absolute_error(y, y_pred_train))

        # 8. PENYIMPANAN MODEL (.pkl) -- VERSIONED, TIDAK MENIMPA MODEL LAMA
        # Setiap hasil retraining disimpan sebagai file baru & dicatat ke
        # models/model_history.json, sehingga model-model sebelumnya tetap
        # tersimpan dan bisa dipilih kembali untuk prediksi.
        model_info = register_new_model(
            model=model,
            features_order=features_order,
            n_rows_train=len(X),
            n_rows_master=len(df_raw),
            params=best_params,
            metrics={"train_rmse": train_rmse, "train_mae": train_mae}
        )

        print(f"[SUCCESS] Model baru berhasil dibuat: {model_info['filename']} "
              f"(versi: {model_info['version_id']})")
        print(f"[INFO] Data latih: {model_info['n_rows_train']} baris | "
              f"Total master data: {model_info['n_rows_master']} baris")
        print(f"[INFO] Model ini otomatis dijadikan model aktif untuk prediksi. "
              f"Riwayat lengkap tersimpan di models/model_history.json")
        return True

    except Exception as e:
        print(f"[ERROR] Terjadi kegagalan pada pipeline: {e}")
        return False

if __name__ == "__main__":
    DEFAULT_FILE = r'E:\airta drafts\PREDIKSI KADAR HB\data\raw\erm_hd.xlsx'
    run_retraining(DEFAULT_FILE)