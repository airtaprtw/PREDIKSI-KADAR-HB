import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sys
import os

# Konfigurasi Path agar bisa memanggil folder notebooks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks')))

from train import run_retraining

st.set_page_config(page_title="Hb Prediction Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }

    /* Menghilangkan tombol +/- pada number input (semua browser) */
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
        appearance: none !important;
        margin: 0 !important;
        display: none !important;
    }
    input[type=number] {
        -moz-appearance: textfield !important;
    }
    /* Menargetkan wrapper Streamlit secara spesifik */
    [data-testid="stNumberInput"] button {
        display: none !important;
    }

    .prediction-card {
        background-color: #2b67ff;
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .stNumberInput > label { font-weight: bold; color: #333; }
    </style>
    """, unsafe_allow_html=True)


# FUNGSI LOGIKA STATUS ANEMIA
def get_anemia_status(hb, jk):
    limit = 13 if jk == 1 else 12
    if hb < 7:
        return "Anemia Berat", "🔴", "Dosis EPO & Transfusi perlu evaluasi segera."
    elif 7 <= hb < 10:
        return "Anemia Sedang", "🟠", "Dosis EPO perlu evaluasi ulang."
    elif 10 <= hb < limit:
        return "Anemia Ringan", "🟡", "Target Hb belum tercapai, jaga nutrisi."
    else:
        return "Normal / Mencapai Target", "🟢", "Kondisi stabil. Pertahankan terapi saat ini."


# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=50)
    st.title("Menu Utama")
    page = st.radio("Pilih Menu", ["Prediction Dashboard", "Retraining System"])
    st.markdown("---")
    st.info("Sistem Prediksi Kadar Hemoglobin Pasien Hemodialisis")


# HALAMAN PREDIKSI

if page == "Prediction Dashboard":
    st.subheader("📋 Identitas Pasien")
    c_id1, c_id2, c_id3 = st.columns(3)

    nama = c_id1.text_input("Nama Pasien", placeholder="Masukkan nama pasien...")
    usia = c_id2.number_input(
        "Usia (Tahun)", min_value=1, max_value=100, step=1,
        value=None, placeholder="Contoh: 45"
    )
    jk = c_id3.selectbox(
        "Jenis Kelamin",
        options=[None, 1, 0],
        format_func=lambda x: "Pilih..." if x is None else ("Laki-laki" if x == 1 else "Perempuan")
    )

    st.markdown("---")

    col_input, col_display = st.columns([1, 1.3])

    with col_input:
        st.subheader("📑 Data Klinis")

        st.write("**Riwayat Hb (g/dL)**")
        h1, h2, h3 = st.columns(3)
        hb_m3 = h1.number_input(
            "Bulan -3", min_value=2.0, max_value=20.0,
            format="%.1f", value=None, placeholder="0.0"
        )
        hb_m2 = h2.number_input(
            "Bulan -2", min_value=2.0, max_value=20.0,
            format="%.1f", value=None, placeholder="0.0"
        )
        hb_m1 = h3.number_input(
            "Bulan -1", min_value=2.0, max_value=20.0,
            format="%.1f", value=None, placeholder="0.0"
        )

        st.write("**Hasil Laboratorium**")
        l1, l2 = st.columns(2)
        leukosit = l1.number_input(
            "Leukosit (cells/µL)", min_value=0, max_value=100000,
            value=None, placeholder="Contoh: 8000"
        )
        trombosit = l2.number_input(
            "Trombosit (cells/µL)", min_value=0, max_value=1500000,
            value=None, placeholder="Contoh: 250000"
        )

        l3, l4 = st.columns(2)
        mcv = l3.number_input(
            "MCV (fL)", min_value=40.0, max_value=150.0,
            format="%.1f", value=None, placeholder="0.0"
        )
        mchc = l4.number_input(
            "MCHC (g/dL)", min_value=20.0, max_value=50.0,
            format="%.1f", value=None, placeholder="0.0"
        )

        st.write("**Terapi**")
        epo = st.radio("Status Pemberian EPO", ["Ya (Rutin)", "Tidak"], horizontal=True)
        epo_val = 1 if epo == "Ya (Rutin)" else 0

        st.markdown("<br>", unsafe_allow_html=True)
        btn_predict = st.button("🚀 PROSES PREDIKSI")

    with col_display:
        if btn_predict:
            # Validasi semua field wajib terisi
            required_fields = {
                "Usia": usia,
                "Jenis Kelamin": jk,
                "Hb Bulan -3": hb_m3,
                "Hb Bulan -2": hb_m2,
                "Hb Bulan -1": hb_m1,
                "Leukosit": leukosit,
                "Trombosit": trombosit,
                "MCV": mcv,
                "MCHC": mchc,
            }
            missing = [k for k, v in required_fields.items() if v is None]

            if missing:
                st.warning(f"⚠️ Mohon lengkapi data berikut: **{', '.join(missing)}**")
            else:
                try:
                    model = joblib.load('models/lgbm_best_model.pkl')

                    # FEATURE ENGINEERING
                    # hb_lag  = Hb bulan lalu (t-1)
                    # hb_delta = selisih Hb bulan lalu vs dua bulan lalu
                    # inflamasi & epo_resist = indikator klinis
                    hb_lag    = hb_m1
                    hb_delta  = hb_m1 - hb_m2
                    inflamasi = (leukosit / 10000) * (trombosit / 150000)
                    epo_resist = epo_val / (inflamasi + 1)

                    # Urutan kolom HARUS sama persis dengan X di train.py:
                    # ['usia', 'jk', 'MCHC', 'MCV', 'leukosit', 'trombosit',
                    #  'hb_lag', 'hb_delta', 'epo_resist']
                    cols_name = [
                        'usia', 'jk', 'MCHC', 'MCV',
                        'leukosit', 'trombosit',
                        'hb_lag', 'hb_delta', 'epo_resist'
                    ]
                    input_data = [[
                        usia, jk, mchc, mcv,
                        leukosit, trombosit,
                        hb_lag, hb_delta, epo_resist
                    ]]
                    input_df = pd.DataFrame(input_data, columns=cols_name)

                    # PREDIKSI TERKINI (M+1)
                    current_pred = model.predict(input_df)[0]

                    # PROYEKSI RECURSIVE 3 BULAN KE DEPAN
                    proj_results = []
                    temp_input   = input_df.copy()
                    last_hb_val  = hb_m1

                    for _ in range(3):
                        p = model.predict(temp_input)[0]
                        proj_results.append(p)

                        temp_input['hb_delta'] = p - last_hb_val
                        temp_input['hb_lag']   = p
                        last_hb_val = p

                    st.markdown(f"""
                        <div class="prediction-card">
                            <p style='margin-bottom:0; font-size: 1.2rem;'>
                                Estimasi Kadar Hb Bulan Depan (M+1)
                            </p>
                            <h1 style='font-size: 4rem; margin: 0;'>
                                {current_pred:.2f}
                                <span style='font-size: 1.5rem;'>g/dL</span>
                            </h1>
                        </div>
                    """, unsafe_allow_html=True)

                    # Status & Saran
                    status, icon, saran = get_anemia_status(current_pred, jk)

                    c_res1, c_res2 = st.columns(2)
                    with c_res1:
                        st.success(f"**Status:** {icon} {status}")
                    with c_res2:
                        st.info(f"**Saran:** {saran}")

                    # Grafik Tren
                    st.write("**Tren Proyeksi Hb 3 Bulan ke Depan**")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[-2, -1, 0], y=[hb_m3, hb_m2, hb_m1],
                        name="Historis",
                        line=dict(color='#2b67ff', width=4)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1, 2, 3], y=[hb_m1] + proj_results,
                        name="Proyeksi (Estimasi)",
                        line=dict(color='#ff9800', dash='dash', width=4)
                    ))
                    fig.update_layout(
                        xaxis_title="Bulan",
                        yaxis_title="g/dL",
                        height=350,
                        margin=dict(l=0, r=0, t=20, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(
                        "⚠️ **Disclaimer:** Akurasi prediksi akan menurun pada bulan ke-2 "
                        "dan ke-3 karena bersifat estimasi recursive."
                    )

                except FileNotFoundError:
                    st.error(
                        "❌ Model belum tersedia. Silakan jalankan **Retraining System** "
                        "terlebih dahulu untuk membuat model."
                    )
                except Exception as e:
                    st.error(f"❌ Gagal memproses prediksi: {e}")


# HALAMAN RETRAINING
elif page == "Retraining System":
    st.title("⚙️ Automated Retraining Model")
    st.write("Unggah data Excel terbaru untuk memperbarui pengetahuan model.")

    file_upload = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"])

    if file_upload:
        if st.button("Mulai Proses Retraining"):
            with st.spinner("Sedang memproses..."):
                temp_path = "temp_data_new.xlsx"
                with open(temp_path, "wb") as f:
                    f.write(file_upload.getbuffer())

                success = run_retraining(temp_path)

                # Hapus file sementara setelah selesai
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if success:
                    st.success("✅ Model LightGBM berhasil diperbarui!")
                    st.balloons()
                else:
                    st.error("❌ Terjadi kesalahan teknis saat retraining. Cek terminal untuk detail error.")
