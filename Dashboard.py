# Dashboard_Explorasi.py
# Exploratory Data Dashboard (clean UI + auto-smart)
# by Muhammad Fahri Zikri

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =============================
# PAGE CONFIG & THEME TUNING
# =============================
st.set_page_config(
    page_title="Exploratory Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS biar rapih & konsisten dengan dashboard final
st.markdown("""
<style>
/* font & spacing */
html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] { width: 340px !important; }

/* cards look */
.card { border-radius: 16px; padding: 1rem 1.2rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); }
.badge { padding: .2rem .6rem; border-radius: 999px; border:1px solid rgba(255,255,255,0.15); font-size:.75rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Exploratory Data Dashboard")

# =============================
# LOAD DATA (GANTI NAMA FILE DI SINI)
# =============================
CSV_PATH = "After_Preparation_TitanicDataset.csv"   # <-- ganti sesuai nama file CSV mentah/hasil prep (tanpa kolom prediksi)
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

try:
    raw_df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Gagal memuat CSV: {e}")
    st.stop()

df = raw_df.copy()

# =============================
# AUTO DETECT TIPE KOLOM
# - kategori: object atau int dengan kardinalitas kecil
# - numerik: int/float
# - usulan target: kolom bernama umum (survived, target, label, kelas, outcome, y)
# =============================
def detect_columns(_df: pd.DataFrame):
    cat_cols = list(_df.select_dtypes(include=["object", "category"]).columns)

    # tambahkan kandidat "kategori numerik" (low-cardinality ints)
    for c in _df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        if _df[c].nunique(dropna=True) <= 20 and c not in cat_cols:
            cat_cols.append(c)

    num_cols = list(_df.select_dtypes(include=["float64", "float32", "int64", "int32", "int16", "int8"]).columns)

    # kandidat target
    lower_cols = {c.lower(): c for c in _df.columns}
    target_candidates = ["survived", "target", "label", "kelas", "class", "outcome", "y"]
    target_col = None
    for cand in target_candidates:
        if cand in lower_cols:
            target_col = lower_cols[cand]
            break

    return sorted(list(dict.fromkeys(cat_cols))), sorted(num_cols), target_col

categorical_cols, numeric_cols, suggested_target = detect_columns(df)

# =============================
# SIDEBAR – FILTER DATA
# aman terhadap NaN (drop untuk range; add opsi "(missing)" utk kategori)
# =============================
with st.sidebar:
    st.header("🔍 Filter Data")

    # Pilih hingga 6 kolom kategori untuk difilter (biar sidebar nggak penuh)
    sel_cat_cols = st.multiselect(
        "Pilih kolom kategori untuk difilter (opsional)",
        options=categorical_cols,
        default=categorical_cols[: min(3, len(categorical_cols))]
    )
    for c in sel_cat_cols:
        vals = df[c].astype("string")
        has_nan = vals.isna() | vals.eq("NaN")
        uniq = sorted(vals.dropna().unique().tolist())
        label_missing = "(missing)"
        choices = [label_missing] + uniq
        picked = st.multiselect(f"{c}:", options=choices, default=choices)  # default pilih semua
        if label_missing not in picked:
            df = df[~df[c].isna()]
        if set(picked) - {label_missing}:
            df = df[df[c].astype("string").isin(list(set(picked) - {label_missing}))]

    # Pilih hingga 3 kolom numerik untuk difilter dengan range
    sel_num_cols = st.multiselect(
        "Pilih kolom numerik untuk difilter (opsional)",
        options=numeric_cols,
        default=numeric_cols[: min(2, len(numeric_cols))]
    )
    for c in sel_num_cols:
        series = df[c]
        if series.dropna().empty:
            st.caption(f"⚠️ Kolom **{c}** semuanya NaN setelah filter di atas — dilewati.")
            continue
        cmin = float(series.dropna().min())
        cmax = float(series.dropna().max())
        vmin, vmax = st.slider(
            f"Rentang {c}:",
            min_value=float(cmin),
            max_value=float(cmax),
            value=(float(cmin), float(cmax))
        )
        df = df[(df[c].fillna(vmin) >= vmin) & (df[c].fillna(vmax) <= vmax)]

    st.download_button(
        "💾 Download Data Terfilter (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="data_terfilter.csv",
        mime="text/csv",
        use_container_width=True
    )

# =============================
# RINGKASAN ATAS (chips)
# =============================
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(f'<div class="card"><span class="badge">Rows</span><h3>{len(df):,}</h3></div>', unsafe_allow_html=True)
with col_b:
    st.markdown(f'<div class="card"><span class="badge">Columns</span><h3>{df.shape[1]}</h3></div>', unsafe_allow_html=True)
with col_c:
    miss_pct = (df.isna().sum().sum() / (df.shape[0]*df.shape[1]))*100 if df.size else 0
    st.markdown(f'<div class="card"><span class="badge">Missing (%)</span><h3>{miss_pct:.2f}%</h3></div>', unsafe_allow_html=True)
with col_d:
    tgt_label = suggested_target if suggested_target else "—"
    st.markdown(f'<div class="card"><span class="badge">Suggested Target</span><h3>{tgt_label}</h3></div>', unsafe_allow_html=True)

st.markdown("---")

# =============================
# TABS
# =============================
t1, t2, t3, t4 = st.tabs(["📋 Data", "📈 Visualisasi", "📊 Statistik", "ℹ️ Info"])

# --- TAB DATA
with t1:
    st.subheader("📋 Data Terfilter")
    st.dataframe(df, use_container_width=True, height=460)

# --- TAB VISUALISASI
with t2:
    st.subheader("📈 Buat Visualisasi")

    # Pilih kolom
    colX, colY = st.columns(2)
    with colX:
        x_col = st.selectbox("Pilih Kolom X:", options=df.columns, index=0 if len(df.columns) else None)
    with colY:
        y_candidates = ["(None)"] + list(df.columns)
        y_col = st.selectbox("Pilih Kolom Y:", options=y_candidates, index=0)

    color_col = st.selectbox("Pilih Kolom Warna (Opsional):", [None] + list(df.columns))

    # Tipe grafik (Auto pilih default terbaik)
    chart_list = ["Auto", "Scatter", "Histogram", "Bar", "Box", "Violin", "Line", "Pie"]
    chart_type = st.selectbox("Pilih Jenis Visualisasi:", chart_list, index=0)

    # logika AUTO
    def is_numeric(col): 
        return pd.api.types.is_numeric_dtype(df[col])

    def pick_auto(x, y):
        if y == "(None)":
            return "Histogram" if is_numeric(x) else "Bar"
        if is_numeric(x) and is_numeric(y):
            return "Scatter"
        if not is_numeric(x) and is_numeric(y):
            return "Bar"
        if is_numeric(x) and not is_numeric(y):
            return "Box"
        return "Bar"

    chosen = chart_type if chart_type != "Auto" else pick_auto(x_col, y_col)

    # Build figure
    fig = None
    if chosen == "Scatter":
        if y_col == "(None)":
            st.warning("Scatter butuh X dan Y.")
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, trendline=None)
    elif chosen == "Histogram":
        fig = px.histogram(df, x=x_col, color=color_col, nbins=40 if is_numeric(x_col) else None, barmode="overlay")
    elif chosen == "Bar":
        if y_col == "(None)":
            # per-category count kalau X kategori
            if not is_numeric(x_col):
                tmp = df[x_col].value_counts(dropna=False).reset_index()
                tmp.columns = [x_col, "count"]
                fig = px.bar(tmp, x=x_col, y="count", color=color_col if color_col in tmp.columns else None)
            else:
                st.info("Bar dengan Y kosong → pilih Y numerik untuk agregasi.")
        else:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col)
    elif chosen == "Box":
        target_y = y_col if y_col != "(None)" else x_col
        target_x = x_col if y_col != "(None)" else None
        if is_numeric(target_y):
            fig = px.box(df, x=target_x, y=target_y, color=color_col, points="suspectedoutliers")
        else:
            st.info("Box plot butuh sumbu Y numerik.")
    elif chosen == "Violin":
        if y_col == "(None)":
            st.info("Violin butuh X/Y: set Y numerik (atau tukar X/Y).")
        else:
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, box=True, points=False)
    elif chosen == "Line":
        if y_col == "(None)":
            st.info("Line plot butuh Y.")
        else:
            fig = px.line(df.sort_values(by=x_col), x=x_col, y=y_col, color=color_col)
    elif chosen == "Pie":
        # gunakan X sebagai kategori; Y opsional sebagai value
        if is_numeric(x_col):
            st.info("Pie memakai kolom kategori sebagai label (pilih X non-numerik).")
        else:
            if y_col != "(None)" and is_numeric(y_col):
                fig = px.pie(df, names=x_col, values=y_col, color=color_col)
            else:
                tmp = df[x_col].value_counts(dropna=False).reset_index()
                tmp.columns = [x_col, "count"]
                fig = px.pie(tmp, names=x_col, values="count", color=x_col)

    if fig is not None:
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=520)
        st.plotly_chart(fig, use_container_width=True)

        # Download image (kaleido)
        try:
            img_bytes = fig.to_image(format="png")
            st.download_button(
                "📷 Download Grafik (PNG)",
                data=img_bytes,
                file_name="visualisasi.png",
                mime="image/png"
            )
        except Exception as e:
            st.caption("💡 Untuk export gambar, install: `pip install -U kaleido`")

# --- TAB STATISTIK
with t3:
    st.subheader("📊 Statistik Singkat")
    colL, colR = st.columns([2,1])
    with colL:
        if not df.select_dtypes(include=[np.number]).empty:
            st.write(df.describe(include=[np.number]).T)
        else:
            st.info("Tidak ada kolom numerik untuk statistik deskriptif.")
    with colR:
        st.write("🧩 Missing by Column")
        miss_tbl = df.isna().mean().sort_values(ascending=False).rename("missing_rate").to_frame()
        st.dataframe((miss_tbl * 100).round(2), use_container_width=True, height=360)

# --- TAB INFO
with t4:
    st.markdown(f"""
    ## ℹ️ Panduan
    Dashboard ini untuk **eksplorasi data** *(sebelum modelling)*.  
    - **Auto-detect** kolom kategori/numerik + usulan target.  
    - **Filter pintar**: kategori & rentang numerik (aman untuk NaN).  
    - **Visualisasi Auto-Smart**: jenis grafik menyesuaikan tipe kolom.  
    - **Download cepat**: data terfilter (CSV) & grafik (PNG).  

    ---
    ### ⚡ Cara Pakai Cepat
    1. Klik link yang sudah disediakan, maka akan automatis dibawa ke website streamlit cloud dimana dataset sudah siap di akses 
    2. Gunakan filter di sidebar untuk menyaring data.  
    3. Pilih jenis visualisasi → hasil otomatis menyesuaikan tipe data.  
    4. Download data atau grafik sesuai kebutuhan.  

    ---
    ### 💡 Tips
    - Tidak perlu repot pilih tipe visualisasi manual — sistem sudah otomatis mendeteksi dan menyesuaikan.  
    - Untuk dataset besar, pastikan file CSV sudah hasil *preparation* agar proses lancar.  
    - Instal **kaleido** sekali saja untuk ekspor grafik ke PNG.  
    - Santai saja, semua fitur sudah siap — tinggal **lihat hasilnya**.  

    """)
st.markdown("---")
st.caption("© 2025 Exploratory Data Dashboard by Muhammad Fahri Zikri")
