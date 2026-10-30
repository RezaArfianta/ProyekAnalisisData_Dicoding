# Proyek Analisis Data: Bike Sharing Dataset

Analisis data penyewaan sepeda (Capital Bikeshare, 2011-2012) beserta dashboard interaktif Streamlit.

## Struktur Folder

```
├── dashboard
│   ├── dashboard.py        # aplikasi Streamlit
│   ├── day_clean.csv       # data harian yang sudah dibersihkan
│   └── hour_clean.csv      # data per jam yang sudah dibersihkan
├── data
│   ├── day.csv
│   └── hour.csv
├── Proyek_Analisis_Data.ipynb
├── README.md
├── requirements.txt
└── url.txt                 # isi dengan URL dashboard setelah deploy
```

## Menjalankan Dashboard di Lokal

1. (Opsional) buat virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
2. Pasang dependensi:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan dashboard:
   ```
   streamlit run dashboard/dashboard.py
   ```
   Dashboard terbuka di http://localhost:8501.

## Menjalankan Notebook

Buka `Proyek_Analisis_Data.ipynb` dan jalankan seluruh sel. Sel terakhir pada bagian Cleaning Data akan
menulis ulang `dashboard/day_clean.csv` dan `dashboard/hour_clean.csv`.
