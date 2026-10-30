from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# Konfigurasi halaman dan gaya grafik
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Bike Sharing", page_icon="🚲", layout="wide")

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_WORK = "#1F77B4"
C_OFF = "#E8833A"
C_MUTED = "#B8BEC6"

DATA_DIR = Path(__file__).parent
WEATHER_ORDER = ["Cerah", "Berawan/Kabut", "Hujan/Salju Ringan", "Hujan Lebat/Badai"]
SEASON_ORDER = ["Spring", "Summer", "Fall", "Winter"]
WEEKDAY_ORDER = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
SEGMENT_LABELS = ["Dini Hari", "Pagi Sibuk", "Siang", "Sore Sibuk", "Malam"]
TEMP_LABELS = ["Dingin (<10°C)", "Sejuk (10-20°C)", "Hangat (20-30°C)", "Panas (>=30°C)"]


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    day = pd.read_csv(DATA_DIR / "day_clean.csv", parse_dates=["dteday"])
    hour = pd.read_csv(DATA_DIR / "hour_clean.csv", parse_dates=["dteday"])

    # Manual grouping jam dan binning suhu (analisis lanjutan tanpa machine learning)
    hour["time_segment"] = pd.cut(hour["hr"], bins=[-1, 5, 9, 15, 19, 23], labels=SEGMENT_LABELS)
    day["temp_bin"] = pd.cut(day["temp_c"], bins=[-np.inf, 10, 20, 30, np.inf],
                             labels=TEMP_LABELS, right=False)
    return day, hour


def fmt(n):
    return f"{n:,.0f}".replace(",", ".")


def pct(x):
    return f"{x:.1f}".replace(".", ",") + "%"


day_all, hour_all = load_data()

# ----------------------------------------------------------------------------
# Sidebar: filter
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Filter Data")
    min_date, max_date = day_all["dteday"].min().date(), day_all["dteday"].max().date()
    date_range = st.date_input("Rentang tanggal", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    seasons = st.multiselect("Musim", SEASON_ORDER, default=SEASON_ORDER)
    weathers = st.multiselect("Cuaca", WEATHER_ORDER[:3], default=WEATHER_ORDER[:3],
                              help="Kategori Hujan Lebat/Badai hanya muncul pada data per jam (3 baris) "
                                   "sehingga disertakan otomatis.")
    st.caption("Filter berlaku untuk semua grafik. Sumber data: Capital Bikeshare 2011-2012.")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
else:
    start = end = date_range[0] if isinstance(date_range, (tuple, list)) else date_range

weather_filter = weathers + ["Hujan Lebat/Badai"]


def apply_filter(df):
    mask = (
        (df["dteday"].dt.date >= start) & (df["dteday"].dt.date <= end)
        & df["season_label"].isin(seasons)
        & df["weather_label"].isin(weather_filter)
    )
    return df[mask]


day_df = apply_filter(day_all)
hour_df = apply_filter(hour_all)

# ----------------------------------------------------------------------------
# Header dan KPI
# ----------------------------------------------------------------------------
st.title("🚲 Dashboard Analisis Bike Sharing")
st.caption("Menjawab tiga pertanyaan bisnis seputar pola jam sibuk, pengaruh cuaca/musim, "
           "dan pertumbuhan penyewaan sepeda tahun 2011-2012.")

if day_df.empty or hour_df.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini. Ubah filter di sidebar.")
    st.stop()

total_rent = day_df["cnt"].sum()
avg_daily = day_df["cnt"].mean()
reg_share = day_df["registered"].sum() / total_rent * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total penyewaan", fmt(total_rent))
k2.metric("Rata-rata per hari", fmt(avg_daily))
k3.metric("Pengguna registered", pct(reg_share))
k4.metric("Jumlah hari", fmt(len(day_df)))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Pola Jam", "2. Cuaca & Musim", "3. Pertumbuhan", "Analisis Lanjutan", "Data"
])

# ----------------------------------------------------------------------------
# Tab 1: Pola jam
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Pada jam berapa penyewaan mencapai puncak di hari kerja vs non-kerja?")

    hourly = hour_df.groupby(["daytype", "hr"])["cnt"].mean().unstack(0)
    workday_hr = hour_df[hour_df["workingday"] == 1].groupby("hr")[["casual", "registered", "cnt"]].mean()

    c1, c2 = st.columns([1.5, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for label, color in [("Hari kerja", C_WORK), ("Non-kerja", C_OFF)]:
            if label in hourly.columns:
                s = hourly[label].dropna()
                ax.plot(s.index, s.values, color=color, linewidth=2.5, marker="o", markersize=4, label=label)
                peak_hr = s.idxmax()
                ax.annotate(f"jam {peak_hr:02d}.00\n({s[peak_hr]:.0f})", xy=(peak_hr, s[peak_hr]),
                            xytext=(0, 18), textcoords="offset points", ha="center",
                            color=color, fontsize=9)
        ax.set_title("Rata-rata penyewaan per jam")
        ax.set_xlabel("Jam dalam sehari")
        ax.set_ylabel("Rata-rata penyewaan per jam")
        ax.set_xticks(range(0, 24, 2))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.legend(frameon=False, loc="upper left")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        seg = (hour_df[hour_df["workingday"] == 1]
               .groupby("time_segment", observed=True)[["registered", "casual"]].mean())
        fig, ax = plt.subplots(figsize=(5, 4.6))
        ax.bar(seg.index.astype(str), seg["registered"], color=C_WORK, label="Registered")
        ax.bar(seg.index.astype(str), seg["casual"], bottom=seg["registered"], color=C_MUTED, label="Casual")
        for i, (r, c) in enumerate(zip(seg["registered"], seg["casual"])):
            ax.text(i, r + c + 5, f"{r / (r + c) * 100:.0f}% reg.", ha="center", fontsize=8)
        ax.set_title("Komposisi pengguna, hari kerja")
        ax.set_ylabel("Rata-rata penyewaan per jam")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        ax.legend(frameon=False, loc="upper left")
        st.pyplot(fig)
        plt.close(fig)

    if len(workday_hr) >= 12:
        m_h = workday_hr.loc[workday_hr.index.isin(range(6, 11)), "cnt"].idxmax()
        e_h = workday_hr.loc[workday_hr.index.isin(range(15, 21)), "cnt"].idxmax()
        m, e = workday_hr.loc[m_h], workday_hr.loc[e_h]
        st.info(
            f"**Hari kerja** memiliki dua puncak: pagi jam **{m_h:02d}.00** ({m['cnt']:.0f} penyewaan/jam, "
            f"{m['registered'] / m['cnt'] * 100:.0f}% registered) dan sore jam **{e_h:02d}.00** "
            f"({e['cnt']:.0f} penyewaan/jam, {e['registered'] / e['cnt'] * 100:.0f}% registered). "
            "Pola ini menandakan penggunaan sebagai sarana komuter."
        )
    if "Non-kerja" in hourly.columns:
        s = hourly["Non-kerja"].dropna()
        st.info(f"**Hari non-kerja** hanya memiliki satu puncak di sekitar jam **{s.idxmax():02d}.00** "
                f"({s.max():.0f} penyewaan/jam).")

# ----------------------------------------------------------------------------
# Tab 2: Cuaca dan musim
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Seberapa besar cuaca dan musim memengaruhi penyewaan harian?")

    weather_stats = (day_df.groupby("weather_label")["cnt"].agg(jumlah_hari="count", rata2="mean")
                     .reindex(WEATHER_ORDER[:3]).dropna())
    season_stats = (day_df.groupby("season_label")["cnt"].agg(jumlah_hari="count", rata2="mean")
                    .reindex(SEASON_ORDER).dropna())

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        has_base = "Cerah" in weather_stats.index
        base = weather_stats.loc["Cerah", "rata2"] if has_base else None
        colors = [C_WORK if idx == "Cerah" else C_MUTED for idx in weather_stats.index]
        bars = ax.bar(weather_stats.index, weather_stats["rata2"], color=colors)
        for bar, (idx, row) in zip(bars, weather_stats.iterrows()):
            label = fmt(row["rata2"])
            if has_base and idx != "Cerah":
                label += f"\n({(row['rata2'] / base - 1) * 100:+.1f}%)".replace(".", ",")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60, label, ha="center", fontsize=9)
            ax.text(bar.get_x() + bar.get_width() / 2, 100, f"n = {int(row['jumlah_hari'])} hari",
                    ha="center", color="white", fontsize=8)
        ax.set_title("Rata-rata penyewaan harian menurut cuaca")
        ax.set_ylabel("Rata-rata penyewaan harian")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: fmt(x)))
        ax.tick_params(axis="x", labelsize=9)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        top = season_stats["rata2"].idxmax()
        colors = [C_WORK if idx == top else C_MUTED for idx in season_stats.index]
        bars = ax.bar(season_stats.index, season_stats["rata2"], color=colors)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60, fmt(bar.get_height()),
                    ha="center", fontsize=9)
        ax.set_title(f"Rata-rata penyewaan harian menurut musim (tertinggi: {top})")
        ax.set_ylabel("Rata-rata penyewaan harian")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: fmt(x)))
        st.pyplot(fig)
        plt.close(fig)

    if has_base and len(weather_stats) > 1:
        parts = [f"**{idx}** {(row['rata2'] / base - 1) * 100:+.1f}%".replace(".", ",")
                 for idx, row in weather_stats.iterrows() if idx != "Cerah"]
        st.info("Dibanding cuaca cerah (" + fmt(base) + " penyewaan/hari), rata-rata penyewaan harian berubah: "
                + ", ".join(parts) + ".")
    elif not has_base:
        st.warning("Persentase penurunan membutuhkan data cuaca Cerah sebagai pembanding. "
                   "Aktifkan kategori Cerah pada filter.")
    if (weather_stats["jumlah_hari"] < 30).any():
        st.caption("Catatan: ada kategori dengan kurang dari 30 hari data, sehingga angkanya perlu dibaca hati-hati.")

# ----------------------------------------------------------------------------
# Tab 3: Pertumbuhan
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Berapa pertumbuhan penyewaan harian 2012 dibanding 2011?")

    monthly = day_df.groupby(["mnth", "year"])["cnt"].mean().unstack("year")
    if not {2011, 2012}.issubset(monthly.columns):
        st.warning("Perbandingan tahunan membutuhkan data 2011 dan 2012. Perluas rentang tanggal di sidebar.")
    else:
        monthly = monthly.dropna(subset=[2011, 2012])
        monthly["growth"] = (monthly[2012] / monthly[2011] - 1) * 100
        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
                       7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}
        labels = [month_names[m] for m in monthly.index]

        # Pertumbuhan keseluruhan dihitung hanya dari bulan yang ada di kedua tahun
        both = day_df[day_df["mnth"].isin(monthly.index)]
        yearly = both.groupby("year")["cnt"].mean()
        overall = (yearly[2012] / yearly[2011] - 1) * 100

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4.6))
            ax.plot(labels, monthly[2011], color=C_OFF, linewidth=2.5, marker="o", label="2011")
            ax.plot(labels, monthly[2012], color=C_WORK, linewidth=2.5, marker="o", label="2012")
            ax.fill_between(labels, monthly[2011], monthly[2012], color=C_WORK, alpha=0.08)
            ax.set_title(f"Penyewaan harian 2012 naik {overall:.0f}% dari 2011")
            ax.set_ylabel("Rata-rata penyewaan harian")
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: fmt(x)))
            ax.legend(frameon=False, loc="upper left")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 4.6))
            g = monthly["growth"]
            colors = [C_WORK if v in (g.max(), g.min()) else C_MUTED for v in g]
            bars = ax.bar(labels, g, color=colors)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{bar.get_height():.0f}%",
                        ha="center", fontsize=8)
            ax.axhline(overall, color="#444444", linestyle="--", linewidth=1,
                       label=f"Rata-rata keseluruhan ({overall:.0f}%)")
            ax.set_title("Pertumbuhan YoY per bulan")
            ax.set_ylabel("Pertumbuhan 2012 vs 2011 (%)")
            ax.set_ylim(0, g.max() * 1.15)
            ax.legend(frameon=False, loc="upper right")
            st.pyplot(fig)
            plt.close(fig)

        st.info(
            f"Rata-rata penyewaan harian naik **{pct(overall)}** (dari {fmt(yearly[2011])} ke {fmt(yearly[2012])}). "
            f"Pertumbuhan tertinggi: **{month_names[g.idxmax()]}** ({pct(g.max())}), "
            f"terendah: **{month_names[g.idxmin()]}** ({pct(g.min())}). "
            "Angka persentase di awal tahun besar sebagian karena basis 2011 masih rendah."
        )

# ----------------------------------------------------------------------------
# Tab 4: Analisis lanjutan
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Analisis Lanjutan: Manual Grouping dan Binning")
    st.caption("RFM dan geospatial tidak digunakan karena dataset tidak memiliki ID pelanggan, nilai transaksi, "
               "maupun data lokasi.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Binning suhu**: rata-rata penyewaan harian per rentang suhu")
        t = day_df.groupby("temp_bin", observed=True)["cnt"].agg(jumlah_hari="count", rata2="mean")
        fig, ax = plt.subplots(figsize=(6, 4.4))
        top_t = t["rata2"].idxmax()
        colors = [C_WORK if idx == top_t else C_MUTED for idx in t.index]
        bars = ax.bar(t.index.astype(str), t["rata2"], color=colors)
        for bar, n in zip(bars, t["jumlah_hari"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60, fmt(bar.get_height()),
                    ha="center", fontsize=9)
            ax.text(bar.get_x() + bar.get_width() / 2, 100, f"n = {n}", ha="center", color="white", fontsize=8)
        ax.set_ylabel("Rata-rata penyewaan harian")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: fmt(x)))
        ax.tick_params(axis="x", rotation=12, labelsize=9)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Manual grouping jam**: rata-rata penyewaan per jam menurut segmen waktu")
        seg_tbl = (hour_df.groupby(["time_segment", "daytype"], observed=True)["cnt"].mean()
                   .unstack("daytype").reindex(SEGMENT_LABELS))
        fig, ax = plt.subplots(figsize=(6, 4.4))
        x = np.arange(len(seg_tbl))
        width = 0.38
        if "Hari kerja" in seg_tbl:
            ax.bar(x - width / 2, seg_tbl["Hari kerja"], width, color=C_WORK, label="Hari kerja")
        if "Non-kerja" in seg_tbl:
            ax.bar(x + width / 2, seg_tbl["Non-kerja"], width, color=C_OFF, label="Non-kerja")
        ax.set_xticks(x)
        ax.set_xticklabels(seg_tbl.index, rotation=12, fontsize=9)
        ax.set_ylabel("Rata-rata penyewaan per jam")
        ax.legend(frameon=False)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Heatmap**: rata-rata penyewaan per jam menurut hari dalam seminggu")
    heat = (hour_df.pivot_table(index="weekday_label", columns="hr", values="cnt", aggfunc="mean")
            .reindex(WEEKDAY_ORDER).dropna(how="all"))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    sns.heatmap(heat, cmap="Blues", ax=ax, cbar_kws={"label": "Rata-rata penyewaan"}, linewidths=0.3)
    ax.set_xlabel("Jam dalam sehari")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------------
# Tab 5: Data
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("Data harian terfilter")
    cols = ["dteday", "season_label", "weather_label", "daytype", "temp_c", "hum_pct",
            "windspeed_kmh", "casual", "registered", "cnt"]
    table = day_df[cols].copy()
    num_cols = table.select_dtypes("number").columns
    table[num_cols] = table[num_cols].round(1)
    st.dataframe(table.rename(columns={
        "dteday": "Tanggal", "season_label": "Musim", "weather_label": "Cuaca", "daytype": "Jenis hari",
        "temp_c": "Suhu (°C)", "hum_pct": "Kelembapan (%)", "windspeed_kmh": "Angin (km/jam)",
        "casual": "Casual", "registered": "Registered", "cnt": "Total"}),
        width="stretch", hide_index=True)
    st.download_button("Unduh data harian (CSV)", day_df[cols].to_csv(index=False).encode("utf-8"),
                       file_name="bike_sharing_harian_terfilter.csv", mime="text/csv")

st.caption("Dibuat dengan Streamlit. Sumber: Bike Sharing Dataset (Capital Bikeshare), 2011-2012.")
