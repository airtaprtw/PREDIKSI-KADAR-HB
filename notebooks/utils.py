import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime


# =====================================================================
# MEKANISME VERSIONING & HISTORY MODEL (untuk Retraining System)
# =====================================================================
# Setiap kali retraining dijalankan, model TIDAK menimpa file lama.
# Model baru disimpan dengan nama unik (versioned) dan dicatat pada
# models/model_history.json, sehingga semua model hasil retraining
# sebelumnya tetap tersimpan dan bisa dipilih kembali untuk prediksi.

MODELS_DIR = 'models'
HISTORY_PATH = os.path.join(MODELS_DIR, 'model_history.json')
ACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, 'active_model.txt')


def load_model_history():
    """Baca seluruh riwayat model yang pernah dilatih (list of dict)."""
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_model_history(history):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def register_new_model(model, features_order, n_rows_train, n_rows_master,
                        params, metrics=None):
    """
    Simpan model hasil retraining sebagai file .pkl BARU dengan nama unik
    (versioned by timestamp) -- model-model sebelumnya tidak dihapus/ditimpa.
    Metadatanya dicatat ke models/model_history.json, dan model ini otomatis
    dijadikan model aktif (dipakai untuk prediksi), namun tetap bisa diganti
    manual ke versi model lain kapan saja.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    timestamp = datetime.now()
    ts_str = timestamp.strftime('%Y%m%d_%H%M%S')
    version_id = f"v{ts_str}"
    filename = f"lgbm_model_{ts_str}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)

    joblib.dump(model, filepath)

    history = load_model_history()
    entry = {
        "version_id": version_id,
        "filename": filename,
        "filepath": filepath,
        "trained_at": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "n_rows_train": int(n_rows_train),
        "n_rows_master": int(n_rows_master),
        "features": list(features_order),
        "params": params,
        "metrics": metrics or {},
    }
    history.append(entry)
    save_model_history(history)

    # Model paling baru otomatis dijadikan model aktif untuk prediksi
    set_active_model_filename(filename)

    return entry


def get_active_model_filename():
    """
    Kembalikan nama file model yang sedang 'aktif' dipakai untuk prediksi.
    Default: model hasil retraining paling terakhir (jika belum pernah
    dipilih manual).
    """
    if os.path.exists(ACTIVE_MODEL_PATH):
        with open(ACTIVE_MODEL_PATH, 'r', encoding='utf-8') as f:
            fname = f.read().strip()
        if fname and os.path.exists(os.path.join(MODELS_DIR, fname)):
            return fname

    history = load_model_history()
    if history:
        return history[-1]['filename']
    return None


def set_active_model_filename(filename):
    """Tandai salah satu versi model sebagai model aktif untuk prediksi."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(ACTIVE_MODEL_PATH, 'w', encoding='utf-8') as f:
        f.write(filename)


def get_file_size_str(filepath):
    """Kembalikan ukuran file dalam format yang mudah dibaca (B/KB/MB/GB)."""
    if not filepath or not os.path.exists(filepath):
        return "N/A"
    size = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == 'B' else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def automated_pipeline(df):
    """
    Fungsi universal untuk mengubah data mentah ERM (erm_hd.xlsx) 
    menjadi data bersih siap model (erm_hd_clean.xlsx).
    """
    df_clean = df.copy()

    # 1. KONVERSI DATATYPE
    cols_numeric = ['eritrosit', 'hematokrit', 'MCHC', 'MCH', 'MCV', 'hemoglobin', 'leukosit', 'trombosit']

    for col in cols_numeric:
        df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.strip().str.replace(',', '.'), errors='coerce')

    df_clean['tgl_lahir'] = pd.to_datetime(df_clean['tgl_lahir'], errors='coerce')
    df_clean['tgl_pemeriksaan'] = pd.to_datetime(df_clean['tgl_pemeriksaan'], errors='coerce')

    # Hitung selisih tahun (Usia)
    df_clean['usia'] = ((df_clean['tgl_pemeriksaan'] - df_clean['tgl_lahir']).dt.days / 365.25).fillna(0).astype(int)

    # Reorder kolom usia
    cols = df_clean.columns.tolist()
    if 'tgl_lahir' in cols:
        idx = cols.index('tgl_lahir')
        cols.insert(idx + 1, cols.pop(cols.index('usia')))
        df_clean = df_clean[cols]

    # 2. HANDLING MISSING VALUES (RATA-RATA PER FITUR)
    for col in cols_numeric:
        rata_rata = df_clean[col].mean()
        df_clean[col] = df_clean[col].fillna(rata_rata)

    cols_int = ['leukosit', 'trombosit']
    for col in cols_int:
        df_clean[col] = df_clean[col].round().astype('Int64')

    # 3. BINERISASI STATUS EPO
    if 'status_epo' in df_clean.columns:
        mapping_epo = {'TIDAK': 0, 'YA': 1}
        df_clean['epo'] = df_clean['status_epo'].map(mapping_epo)
        
        cols = df_clean.columns.tolist()
        idx_epo = cols.index('status_epo')
        cols.insert(idx_epo + 1, cols.pop(cols.index('epo')))
        df_clean = df_clean[cols]

    # 4. BINERISASI JENIS KELAMIN
    if 'jenis_kelamin' in df_clean.columns:
        mapping_jk = {'P': 0, 'L': 1}
        df_clean['jk'] = df_clean['jenis_kelamin'].map(mapping_jk)

        cols = df_clean.columns.tolist()
        idx_jk = cols.index('jenis_kelamin')
        cols.insert(idx_jk + 1, cols.pop(cols.index('jk')))
        df_clean = df_clean[cols]
    
    # Drop kolom asli yang sudah dibinerisasi/dikonversi
    cols_to_drop = [c for c in ['tgl_lahir', 'status_epo', 'jenis_kelamin'] if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    # 5. AGGREGATION PER BULAN
    cols_to_mean = ['usia', 'jk', 'eritrosit', 'hematokrit', 'MCHC', 'MCH', 'MCV', 
                    'hemoglobin', 'leukosit', 'trombosit', 'epo']
    
    available_cols = [c for c in cols_to_mean if c in df_clean.columns]

    df_clean = (df_clean.groupby('id_pasien')
                .resample('MS', on='tgl_pemeriksaan')[available_cols]
                .mean()
                .dropna()
                .reset_index())

    cols_to_int = ['usia', 'jk', 'leukosit', 'trombosit', 'epo']
    for col in cols_to_int:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].round().astype('Int64')

    # PRUNING: Hapus ID yang datanya < 3 bulan
    counts = df_clean['id_pasien'].value_counts()
    id_yang_dihapus = counts[counts < 3].index.tolist()
    df_clean = df_clean[~df_clean['id_pasien'].isin(id_yang_dihapus)].copy()

    # RESTORASI DATA (Bulan Loncat)
    cols_statis = ['usia', 'jk', 'epo']
    cols_hematologi = ['hemoglobin', 'hematokrit', 'eritrosit', 'MCV', 'MCH', 'MCHC', 'leukosit', 'trombosit']
    all_cols = [c for c in (cols_statis + cols_hematologi) if c in df_clean.columns]

    df_complete = (df_clean.groupby('id_pasien')
                    .resample('MS', on='tgl_pemeriksaan')[all_cols]
                    .mean()
                    .reset_index())

    for col in cols_statis:
        if col in df_complete.columns:
            df_complete[col] = df_complete.groupby('id_pasien')[col].ffill().bfill()

    for col in cols_hematologi:
        if col in df_complete.columns:
            df_complete[col] = df_complete[col].fillna(df_complete.groupby('id_pasien')[col].transform('mean'))

    cols_to_int_final = ['usia', 'jk', 'epo', 'leukosit', 'trombosit']
    for col in cols_to_int_final:
        if col in df_complete.columns:
            df_complete[col] = df_complete[col].round().astype('Int64')

    # PROTEKSI: Pastikan urutan kolom selalu sama untuk Model
    final_features = ['id_pasien', 'tgl_pemeriksaan', 'usia', 'jk', 'eritrosit', 'hematokrit', 
                      'MCHC', 'MCH', 'MCV', 'hemoglobin', 'leukosit', 'trombosit', 'epo']
    df_complete = df_complete[final_features]

    df_complete = df_complete.sort_values(['id_pasien', 'tgl_pemeriksaan']).reset_index(drop=True)
    return df_complete