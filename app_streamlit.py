"""
Visualizador de señales EEG (Stroop) - Versión Streamlit
=========================================================
Adaptación completa y deployable de Demo_processing.py a Streamlit.
Incluye:
  1. Trim automático de los primeros 3 segundos.
  2. Autoescalado robusto por canal.
  3. Navegación por ventanas temporales configurables (epochs / slider).
  4. Selector de señales crudas o filtradas por bandas (Delta, Theta, Alpha, Beta).
  5. Detección y visualización de ráfagas theta fásicas en Fz con umbral configurable.
  6. Análisis de PSD (Welch) e índices de carga cognitiva (Theta/Alpha, Beta/Alpha, Engagement).
  7. Biomarcadores dinámicos (theta tónico, alfa tónico, FAA, tasa de ráfagas).
  8. Comparación estadística entre niveles (Stroop) tanto en bandas relativas como en biomarcadores.
  9. Soporte para carga de archivos propios (.txt) además de escaneo local.
"""

import os
import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import streamlit as st

# ----------------------------------------------------------------------------
# Configuración general y constantes
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Visualizador EEG - Stroop",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_DIR = os.path.join("raw_data", "raw_data", "Stroop_Data")
DEFAULT_FS = 250.0  # Hz (OpenBCI Cyton)
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'C2']
FRONTAL = ['Fp1', 'Fp2', 'F7', 'F3', 'FZ', 'F4', 'F8']
BANDS = {
    "Delta": (0, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
}
BAND_EDGES = [0, 4, 8, 13, 30]
BAND_COLORS = {
    "Delta": "#9e9e9e",
    "Theta": "#1f77b4",
    "Alpha": "#2ca02c",
    "Beta": "#ff7f0e"
}
LEVEL_PREFIX = {
    "Natural": 'n',
    "Low level": 'l',
    "Mid level": 'm',
    "High level": 'h'
}

THETA_CH = 'FZ'
ALPHA_PREF = ['O1', 'O2', 'Pz']
ALPHA_FALLBACK = ['C2']
FAA_L, FAA_R = 'F3', 'F4'


# ----------------------------------------------------------------------------
# Funciones de Procesamiento y Pipeline
# ----------------------------------------------------------------------------
def numeric_sort_key(filename):
    m = re.search(r'(\d+)\.txt$', filename)
    return int(m.group(1)) if m else 0


def load_eeg(source):
    """
    Lee un archivo o buffer .txt (col 0 = índice de muestra, cols 1-8 = canales).
    Devuelve un numpy array de (n, 8).
    """
    if isinstance(source, (str, os.PathLike)):
        df = pd.read_csv(source, delimiter=",", header=None, comment='%',
                         skipinitialspace=True, engine="python", on_bad_lines="skip")
    else:
        # UploadedFile de streamlit
        df = pd.read_csv(source, delimiter=",", header=None, comment='%',
                         skipinitialspace=True, engine="python", on_bad_lines="skip")
    df = df.iloc[:, 1:9].apply(pd.to_numeric, errors="coerce").dropna()
    return df.to_numpy(dtype=float)


def repair_artifacts(x, fs, thr_uv=150.0, pad_s=0.15, max_frac=0.25):
    """
    Parpadeos, spikes y deslizamientos de electrodo (todos los canales).
    Las muestras que superan +-thr_uv (con margen pad_s) se reemplazan por interpolación lineal.
    """
    x = x.copy()
    struct_len = int(pad_s * fs) * 2 + 1
    bad = ndimage.binary_dilation(np.abs(x) > thr_uv, structure=np.ones((struct_len, 1)))
    idx = np.arange(len(x))
    for ch in range(x.shape[1]):
        b = bad[:, ch]
        if b.any() and b.mean() <= max_frac:
            x[b, ch] = np.interp(idx[b], idx[~b], x[~b, ch])
    return x


def preprocess(x, fs, bandpass=True, notch=None, trim_seconds=3.0):
    """Pipeline de limpieza de EEG: trim -> centrado -> filtros -> reparar artefactos."""
    trim_samples = int(trim_seconds * fs)
    x = x[trim_samples:] if len(x) > trim_samples else x

    # Centrado: remueve offset DC
    x = x - np.median(x, axis=0)

    nyq = fs / 2.0
    if bandpass and len(x) > 100:
        high_cut = min(40.0, nyq * 0.95)
        sos = signal.butter(4, [1.0, high_cut], btype="band", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x, axis=0)

    if notch and notch < nyq and len(x) > 100:
        b, a = signal.iirnotch(notch, Q=30, fs=fs)
        x = signal.filtfilt(b, a, x, axis=0)

    return repair_artifacts(x, fs)


def compute_psd(x, fs):
    """PSD de Welch por canal. Devuelve f, P (ch, nf) en uV^2/Hz."""
    nperseg = int(min(len(x), 2 * fs))
    f, P = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                        window="hann", axis=0, detrend="constant")
    return f, P.T


def band_powers(f, P):
    """Potencia absoluta por banda. P: (ch, nf) -> dict banda -> (ch,)."""
    df = f[1] - f[0] if len(f) > 1 else 1.0
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[name] = P[:, m].sum(axis=1) * df
    return out


def cognitive_indices(bp):
    eps = 1e-12
    th, al, be = bp["Theta"], bp["Alpha"], bp["Beta"]
    return {
        "Theta/Alpha": th / (al + eps),
        "Beta/Alpha": be / (al + eps),
        "Engagement β/(α+θ)": be / (al + th + eps),
    }


def ch_index(name):
    for i, n in enumerate(CHANNEL_NAMES):
        if n.upper() == name.upper():
            return i
    return None


def alpha_channels():
    pref = [ch_index(n) for n in ALPHA_PREF if ch_index(n) is not None]
    if pref:
        return pref, "/".join(ALPHA_PREF)
    fb = [ch_index(n) for n in ALPHA_FALLBACK if ch_index(n) is not None]
    return fb, "/".join(ALPHA_FALLBACK) + " (proxy)"


def band_filter(x, fs, lo, hi):
    hi = min(hi, fs / 2 * 0.95)
    if lo <= 0:
        sos = signal.butter(4, hi, btype="low", fs=fs, output="sos")
    else:
        sos = signal.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def detect_theta_bursts(fz, fs, ratio=2.5, min_dur=0.15, max_dur=3.0, merge_gap=0.1):
    fz = np.asarray(fz, dtype=float)
    if len(fz) < 100:
        z = np.zeros(len(fz))
        return z, z, z + 1e-9, []
    th = band_filter(fz, fs, *BANDS["Theta"])
    env = np.abs(signal.hilbert(th))
    k = max(1, int(0.25 * fs))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    w = max(int(10 * fs), 3)
    base = pd.Series(env).rolling(w, center=True, min_periods=max(1, w // 4)).median().to_numpy()
    base = np.maximum(base, 1e-9)
    mask = env > ratio * base
    d = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    regs = []
    for s0, e0 in zip(starts, ends):
        if regs and (s0 - regs[-1][1]) / fs < merge_gap:
            regs[-1][1] = e0
        else:
            regs.append([s0, e0])
    bursts = [(s0 / fs, e0 / fs, float(env[s0:e0].max()))
              for s0, e0 in regs if min_dur <= (e0 - s0) / fs <= max_dur]
    return th, env, base, bursts


def _metrics_from_bp(bp):
    eps = 1e-12
    th_i = ch_index(THETA_CH)
    a_idx, _ = alpha_channels()
    l, r = ch_index(FAA_L), ch_index(FAA_R)
    theta = float(bp["Theta"][th_i]) if th_i is not None else np.nan
    alpha = float(np.mean(bp["Alpha"][a_idx])) if a_idx else np.nan
    faa = (float(np.log(bp["Alpha"][r] + eps) - np.log(bp["Alpha"][l] + eps))
           if l is not None and r is not None else np.nan)
    return theta, alpha, faa


def biomarker_summary(x, fs, bursts, dur):
    f, P = compute_psd(x, fs)
    bp = band_powers(f, P)
    tot = sum(bp.values())
    theta, alpha, faa = _metrics_from_bp(bp)
    th_i = ch_index(THETA_CH)
    return {
        "theta_fz": theta,
        "theta_fz_rel": float(bp["Theta"][th_i] / tot[th_i] * 100) if th_i is not None else np.nan,
        "alpha": alpha,
        "faa": faa,
        "n_bursts": len(bursts),
        "burst_rate": len(bursts) / (dur / 60.0) if dur > 0 else np.nan,
        "burst_dur": float(np.mean([b[1] - b[0] for b in bursts])) if bursts else np.nan,
        "burst_amp": float(np.mean([b[2] for b in bursts])) if bursts else np.nan,
    }


def window_biomarkers(x, fs, bursts, win_s, step_s):
    n, win, step = len(x), int(win_s * fs), max(1, int(step_s * fs))
    starts = np.arange(0, n - win + 1, step)
    bstart = np.array([b[0] for b in bursts]) if bursts else np.array([])
    out = {"t": [], "theta": [], "alpha": [], "faa": [], "rate": []}
    for s0 in starts:
        f, P = compute_psd(x[s0:s0 + win], fs)
        theta, alpha, faa = _metrics_from_bp(band_powers(f, P))
        t0 = s0 / fs
        cnt = int(np.sum((bstart >= t0) & (bstart < t0 + win_s))) if len(bstart) else 0
        out["t"].append(t0 + win_s / 2)
        out["theta"].append(theta)
        out["alpha"].append(alpha)
        out["faa"].append(faa)
        out["rate"].append(cnt / (win_s / 60.0))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def robust_limits(y, robust=True, pad=0.12):
    lo, hi = (np.percentile(y, [0.5, 99.5]) if robust else (y.min(), y.max()))
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    m = (hi - lo) * pad
    return lo - m, hi + m


# ----------------------------------------------------------------------------
# Cache para acelerar la carga y procesamiento
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess_file(file_path_or_bytes, fs, bandpass, notch, trim_seconds=3.0):
    raw = load_eeg(file_path_or_bytes)
    cleaned = preprocess(raw, fs, bandpass=bandpass, notch=notch, trim_seconds=trim_seconds)
    return cleaned


# ----------------------------------------------------------------------------
# Sidebar / Configuración de entrada
# ----------------------------------------------------------------------------
st.sidebar.title("🧠 EEG Control & Filtros")

data_source_mode = st.sidebar.radio("Origen de datos:", ["Carpeta del dataset", "Subir archivo manual"])

levels_dict = {}
selected_file_path = None
selected_file_name = None
uploaded_content = None

if data_source_mode == "Carpeta del dataset":
    dataset_dir = st.sidebar.text_input("Ruta de la carpeta Stroop:", value=DEFAULT_DIR)
    if os.path.isdir(dataset_dir):
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.txt')]
        for lvl, p in LEVEL_PREFIX.items():
            matched = sorted([f for f in files if f.lower().startswith(p)], key=numeric_sort_key)
            if matched:
                levels_dict[lvl] = matched

        if levels_dict:
            sel_level = st.sidebar.selectbox("Nivel de dificultad:", list(levels_dict.keys()))
            available_files = levels_dict[sel_level]
            selected_file_name = st.sidebar.selectbox("Archivo de registro:", available_files)
            selected_file_path = os.path.join(dataset_dir, selected_file_name)
        else:
            st.sidebar.warning("No se encontraron archivos .txt con prefijos n*, l*, m*, h* en la carpeta.")
    else:
        st.sidebar.error(f"Carpeta no encontrada: {dataset_dir}")
else:
    uploaded = st.sidebar.file_uploader("Sube un archivo .txt de EEG:", type=["txt"])
    if uploaded is not None:
        uploaded_content = uploaded
        selected_file_name = uploaded.name

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de Adquisición y Filtro")
fs = st.sidebar.number_input("Frecuencia de muestreo (Hz):", min_value=50.0, max_value=2000.0, value=DEFAULT_FS, step=10.0)
trim_seconds = st.sidebar.number_input("Trim inicial (segundos):", min_value=0.0, max_value=30.0, value=3.0, step=0.5,
                                      help="Descarta los primeros segundos para estabilizar la señal y eliminar transitorios.")

bandpass = st.sidebar.checkbox("Filtro Pasa-banda 1-40 Hz", value=True)
notch_choice = st.sidebar.selectbox("Filtro Notch (Red eléctrica):", ["Sin notch", "Notch 50 Hz", "Notch 60 Hz"])
notch_val = 50.0 if "50" in notch_choice else 60.0 if "60" in notch_choice else None

st.sidebar.markdown("---")
st.sidebar.subheader("Visualización y Umbrales")
band_show = st.sidebar.selectbox("Señal a graficar:", ["Cruda (filtrada)"] + list(BANDS.keys()))
burst_ratio = st.sidebar.slider("Umbral ráfaga theta Fz (× base):", min_value=1.5, max_value=6.0, value=2.5, step=0.1)
autoscale = st.sidebar.checkbox("Autoescalar canal", value=True)
robust_scale = st.sidebar.checkbox("Escala robusta (percentiles 0.5 - 99.5)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Selección de Canales")
visible_channels = []
cols_ch = st.sidebar.columns(2)
for i, ch_name in enumerate(CHANNEL_NAMES):
    col = cols_ch[i % 2]
    if col.checkbox(ch_name, value=True, key=f"ch_{ch_name}"):
        visible_channels.append(ch_name)

if not visible_channels:
    visible_channels = CHANNEL_NAMES

# ----------------------------------------------------------------------------
# Flujo Principal
# ----------------------------------------------------------------------------
target_source = selected_file_path if data_source_mode == "Carpeta del dataset" else uploaded_content

if target_source is None:
    st.info("👋 Selecciona una carpeta válida con archivos .txt o sube un archivo desde la barra lateral para comenzar.")
    st.stop()

# Cargar y preprocesar datos
try:
    if isinstance(target_source, (str, os.PathLike)):
        data = load_and_preprocess_file(target_source, fs, bandpass, notch_val, trim_seconds=trim_seconds)
    else:
        # Buffer de memoria
        bytes_data = io.BytesIO(target_source.getvalue())
        data = preprocess(load_eeg(bytes_data), fs, bandpass=bandpass, notch=notch_val, trim_seconds=trim_seconds)
except Exception as e:
    st.error(f"Error cargando y procesando el archivo: {e}")
    st.stop()

total_samples = len(data)
total_dur = total_samples / fs
th_idx = ch_index(THETA_CH)

# Detección de ráfagas theta Fz
theta_fz, env_fz, base_fz, bursts = detect_theta_bursts(data[:, th_idx], fs, ratio=burst_ratio)

# Filtrado por banda en caso de solicitar banda específica
if band_show != "Cruda (filtrada)":
    display_signal = band_filter(data, fs, *BANDS[band_show])
else:
    display_signal = data

# ----------------------------------------------------------------------------
# Métricas Rápidas en Cabecera
# ----------------------------------------------------------------------------
st.title(f"📊 Análisis de EEG Stroop: `{selected_file_name}`")
mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
mcol1.metric("Duración (post-trim)", f"{total_dur:.1f} s")
mcol2.metric("Muestras", f"{total_samples:,}")
mcol3.metric("Frecuencia", f"{fs:.0f} Hz")
mcol4.metric("Ráfagas θ (Fz)", f"{len(bursts)}")
burst_rate_total = len(bursts) / (total_dur / 60.0) if total_dur > 0 else 0
mcol5.metric("Tasa ráfagas", f"{burst_rate_total:.1f} /min")

st.markdown("---")

# ----------------------------------------------------------------------------
# Pestañas de la Aplicación
# ----------------------------------------------------------------------------
tab_signals, tab_psd, tab_biomarkers, tab_comparison = st.tabs([
    "📈 Señales en el Tiempo (Epochs)",
    "⚡ PSD y Carga Cognitiva",
    "🔬 Biomarcadores del Scope",
    "📊 Comparación entre Niveles"
])

# ----------------------------------------------------------------------------
# PESTAÑA 1: Señales en el tiempo
# ----------------------------------------------------------------------------
with tab_signals:
    st.subheader("Navegación temporal de las señales EEG")
    ctrl_col1, ctrl_col2 = st.columns([1, 3])
    
    with ctrl_col1:
        win_opt = st.selectbox(
            "Duración de la ventana:",
            ["1 s", "2 s", "5 s", "10 s", "20 s", "30 s", "60 s", "Todo"],
            index=3
        )
        if win_opt == "Todo":
            win_sec = total_dur
        else:
            win_sec = min(float(win_opt.split()[0]), total_dur)

    max_start = max(0.0, total_dur - win_sec)
    
    with ctrl_col2:
        if max_start > 0:
            start_sec = st.slider("Inicio de la ventana (s):", min_value=0.0, max_value=float(max_start), value=0.0, step=max(0.1, win_sec / 10.0))
        else:
            start_sec = 0.0
            st.caption("Mostrando registro completo.")

    i0 = int(round(start_sec * fs))
    i1 = min(total_samples, i0 + int(round(win_sec * fs)))
    t_win = np.arange(i0, i1) / fs

    if len(t_win) < 2:
        st.warning("El rango de tiempo seleccionado es demasiado corto.")
    else:
        active_indices = [ch_index(name) for name in visible_channels if ch_index(name) is not None]
        n_plots = len(active_indices)
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, max(2.5, 1.2 * n_plots)), sharex=True, squeeze=False)
        axes = axes[:, 0]
        
        for ax, ch in zip(axes, active_indices):
            y = display_signal[i0:i1, ch]
            ch_name = CHANNEL_NAMES[ch]
            ax.plot(t_win, y, color="#1f4e9c", lw=0.8)
            ax.set_ylabel(f"{ch_name}\n(µV)", fontsize=8, fontweight="bold")
            ax.grid(True, ls="--", alpha=0.4)
            ax.tick_params(labelsize=8)
            
            if autoscale:
                ax.set_ylim(*robust_limits(y, robust=robust_scale))
            ax.set_xlim(t_win[0], t_win[-1])
            
            # Si estamos viendo Theta y es canal FZ, sombrear ráfagas
            if band_show == "Theta" and ch == th_idx:
                for b0, b1, _ in bursts:
                    if b1 > t_win[0] and b0 < t_win[-1]:
                        ax.axvspan(max(b0, t_win[0]), min(b1, t_win[-1]), color="#d62728", alpha=0.25)

        axes[-1].set_xlabel("Tiempo (s)", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ----------------------------------------------------------------------------
# PESTAÑA 2: PSD y Carga Cognitiva
# ----------------------------------------------------------------------------
with tab_psd:
    st.subheader("Densidad Espectral de Potencia (Welch) y Carga Cognitiva")
    psd_c1, psd_c2, psd_c3 = st.columns([1, 1, 1])
    with psd_c1:
        seg_choice = st.selectbox("Segmento a analizar:", ["Epoch actual (ventana visible)", "Registro completo"])
    with psd_c2:
        scale_choice = st.selectbox("Escala eje Y:", ["Logarítmico (dB)", "Lineal"])
    with psd_c3:
        avg_only = st.checkbox("Solo curva promedio de canales", value=False)

    if seg_choice == "Registro completo":
        seg_data = data
    else:
        seg_data = data[i0:i1]

    if len(seg_data) < int(fs):
        st.warning("El segmento es menor a 1 segundo; no es suficiente para estimar PSD.")
    else:
        active_indices = [ch_index(name) for name in visible_channels if ch_index(name) is not None]
        seg_vis = seg_data[:, active_indices]
        names_vis = [CHANNEL_NAMES[i] for i in active_indices]

        f, P = compute_psd(seg_vis, fs)
        m = (f > 0) & (f <= 30)
        log_scale = scale_choice.startswith("Log")

        fig_psd = plt.figure(figsize=(12, 7))
        gs = fig_psd.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1.5, 1])
        ax_psd = fig_psd.add_subplot(gs[0, :])
        ax_bar = fig_psd.add_subplot(gs[1, 0])
        ax_txt = fig_psd.add_subplot(gs[1, 1])

        # Curvas PSD
        for name, (lo, hi) in BANDS.items():
            ax_psd.axvspan(lo, hi, color=BAND_COLORS[name], alpha=0.10)
            ax_psd.text((lo + hi) / 2, 0.98, name, transform=ax_psd.get_xaxis_transform(),
                        ha="center", va="top", fontsize=9, color=BAND_COLORS[name], fontweight="bold")

        mean_psd = P.mean(axis=0)
        if not avg_only:
            for k, n in enumerate(names_vis):
                y_val = 10 * np.log10(P[k, m] + 1e-20) if log_scale else P[k, m]
                ax_psd.plot(f[m], y_val, lw=0.9, alpha=0.75, label=n)

        y_avg = 10 * np.log10(mean_psd[m] + 1e-20) if log_scale else mean_psd[m]
        ax_psd.plot(f[m], y_avg, color="k", lw=2.2, label="Promedio")
        ax_psd.set_xlim(0, 30)
        ax_psd.set_xticks(BAND_EDGES)
        ax_psd.set_xticklabels([str(e) for e in BAND_EDGES])
        ax_psd.set_xlabel("Frecuencia (Hz)")
        ax_psd.set_ylabel("PSD (dB re 1 µV²/Hz)" if log_scale else "PSD (µV²/Hz)")
        ax_psd.set_title(f"PSD de Welch ({len(seg_data) / fs:.1f} s analizados)")
        ax_psd.grid(True, ls="--", alpha=0.4)
        ax_psd.legend(ncol=min(len(names_vis) + 1, 9), fontsize=8, loc="lower left")

        # Barras de potencia relativa
        bp = band_powers(f, P)
        tot = sum(v for v in bp.values())
        rel = {k: float(np.mean(v / tot)) * 100 for k, v in bp.items()}

        ax_bar.bar(list(rel.keys()), list(rel.values()), color=[BAND_COLORS[k] for k in rel])
        for idx_b, v in enumerate(rel.values()):
            ax_bar.text(idx_b, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
        ax_bar.set_ylabel("Potencia relativa (%)")
        ax_bar.set_title("Distribución de potencia (0-30 Hz)")
        ax_bar.set_ylim(0, max(rel.values()) * 1.25)
        ax_bar.grid(True, axis="y", ls="--", alpha=0.4)

        # Índices cognitivos
        idx_cog = cognitive_indices(bp)
        ax_txt.axis("off")
        lines = ["ÍNDICES DE CARGA COGNITIVA", "(promedio canales visibles)", ""]
        for k, v in idx_cog.items():
            lines.append(f"{k}: {np.mean(v):.2f}")
        fr = [i for i, n in enumerate(names_vis) if n in FRONTAL]
        if fr:
            lines += ["", f"θ frontal: {np.mean(bp['Theta'][fr]):.2f} µV²",
                      f"α: {np.mean(bp['Alpha']):.2f} µV²"]
        ax_txt.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da"))

        fig_psd.tight_layout()
        st.pyplot(fig_psd)
        plt.close(fig_psd)

# ----------------------------------------------------------------------------
# PESTAÑA 3: Biomarcadores del Scope
# ----------------------------------------------------------------------------
with tab_biomarkers:
    st.subheader("Biomarcadores de Neurofeedback & Fatiga Cognitiva")
    bm_seg_opt = st.radio("Segmento para biomarcadores:", ["Epoch actual", "Registro completo"], horizontal=True)
    if bm_seg_opt == "Registro completo":
        s_i0, s_i1 = 0, total_samples
    else:
        s_i0, s_i1 = i0, i1

    bm_t0, bm_t1 = s_i0 / fs, s_i1 / fs
    bm_dur = bm_t1 - bm_t0

    if bm_dur < 4.0:
        st.warning("Se requieren al menos 4 segundos para calcular biomarcadores confiables.")
    else:
        bm_bursts = [b for b in bursts if b[0] >= bm_t0 and b[1] <= bm_t1]
        bm_rel_bursts = [(b[0] - bm_t0, b[1] - bm_t0, b[2]) for b in bm_bursts]
        seg_bm = data[s_i0:s_i1]
        a_idx, a_label = alpha_channels()

        fig_bm = plt.figure(figsize=(12, 8))
        gs_bm = fig_bm.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], width_ratios=[1.5, 1])

        # Gráfico superior: Ráfagas en Fz
        ax0 = fig_bm.add_subplot(gs_bm[0, :])
        t_plot = np.arange(s_i0, s_i1) / fs
        ax0.plot(t_plot, theta_fz[s_i0:s_i1], color=BAND_COLORS["Theta"], lw=0.8, label="Theta Fz (4-8 Hz)")
        ax0.plot(t_plot, env_fz[s_i0:s_i1], color="k", lw=1.1, label="Envolvente")
        ax0.plot(t_plot, burst_ratio * base_fz[s_i0:s_i1], "--", color="gray", lw=1.1, label=f"Umbral ({burst_ratio:g}× base)")
        for k_b, b in enumerate(bm_bursts):
            ax0.axvspan(b[0], b[1], color="#d62728", alpha=0.28, label="Ráfaga fásica (FM-theta)" if k_b == 0 else None)
        ax0.set_xlim(bm_t0, bm_t1)
        ax0.set_ylim(*robust_limits(theta_fz[s_i0:s_i1], False))
        ax0.set_xlabel("Tiempo (s)")
        ax0.set_ylabel("µV")
        ax0.set_title(f"Theta Fz - Ráfagas detectadas en el segmento: {len(bm_bursts)}", fontsize=10)
        ax0.grid(True, ls="--", alpha=0.4)
        ax0.legend(ncol=4, fontsize=8, loc="upper right")

        # Dinámica temporal en ventanas deslizantes
        win_s = float(np.clip(bm_dur / 6.0, 4.0, 10.0))
        step_s = win_s / 2.0
        enough = bm_dur >= 8.0

        ax1 = fig_bm.add_subplot(gs_bm[1, 0])
        ax2 = fig_bm.add_subplot(gs_bm[2, 0], sharex=ax1)
        ax3 = fig_bm.add_subplot(gs_bm[1, 1])
        ax4 = fig_bm.add_subplot(gs_bm[2, 1])
        ax3.axis("off")
        ax4.axis("off")

        if enough:
            wb = window_biomarkers(seg_bm, fs, bm_rel_bursts, win_s, step_s)
            unit, ulabel = (60.0, "min") if bm_dur >= 120.0 else (1.0, "s")
            tc = (bm_t0 + wb["t"]) / unit
            nb_ = max(1, int(round(0.2 * len(tc))))

            def norm(v):
                base = np.nanmean(v[:nb_])
                return v / base if np.isfinite(base) and base > 0 else v * np.nan

            th_n, al_n = norm(wb["theta"]), norm(wb["alpha"])
            ax1.plot(tc, th_n, color=BAND_COLORS["Theta"], lw=2.2, label="Theta tónico")
            ax1.plot(tc, al_n, color=BAND_COLORS["Alpha"], lw=2.2, label="Alfa tónico")
            ax1.axhline(1.0, color="gray", ls=":", lw=1)
            ax1.set_ylabel("Relativo a base")
            ax1.legend(fontsize=8, loc="best")
            ax1.grid(True, ls="--", alpha=0.4)
            ax1.tick_params(labelbottom=False)

            ax2.bar(tc, wb["rate"], width=step_s / unit * 0.9, color="#d62728", alpha=0.45)
            ax2.set_ylabel("Ráfagas/min", color="#d62728")
            ax2b = ax2.twinx()
            ax2b.plot(tc, wb["faa"], color="#1f77b4", lw=2, marker="o", ms=3)
            ax2b.axhline(0, color="gray", ls=":", lw=1)
            ax2b.set_ylabel("FAA", color="#1f77b4")
            ax2.set_xlabel(f"Tiempo ({ulabel})")
            ax2.grid(True, ls="--", alpha=0.4)

        # Resumen numérico
        s = biomarker_summary(seg_bm, fs, bm_rel_bursts, bm_dur)
        def fmt(v, f="{:.2f}"):
            return f.format(v) if np.isfinite(v) else "n/d"

        L = [
            f"RESUMEN BIOMARCADORES",
            f"Segmento: {bm_dur:.1f} s post-trim",
            "-" * 28,
            f"θ tónico Fz:        {fmt(s['theta_fz'])} µV²",
            f"θ Fz relativo:     {fmt(s['theta_fz_rel'])} %",
            f"α tónico ({a_label}): {fmt(s['alpha'])} µV²",
            f"FAA (F4 vs F3):     {fmt(s['faa'])}",
            f"Ráfagas θ (Fz):     {s['n_bursts']}",
            f"Tasa ráfagas:       {fmt(s['burst_rate'], '{:.1f}')} /min",
            f"Duración media:     {fmt(s['burst_dur'])} s",
            f"Amplitud media:     {fmt(s['burst_amp'])} µV",
        ]
        ax3.text(0.05, 0.95, "\n".join(L), va="top", fontsize=9, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da"))

        fig_bm.tight_layout()
        st.pyplot(fig_bm)
        plt.close(fig_bm)

# ----------------------------------------------------------------------------
# PESTAÑA 4: Comparación entre Niveles Stroop
# ----------------------------------------------------------------------------
with tab_comparison:
    st.subheader("Comparación de Niveles de Carga (Stroop)")
    if not levels_dict or data_source_mode != "Carpeta del dataset":
        st.info("Para comparar niveles, asegúrate de haber seleccionado 'Carpeta del dataset' y que contenga los niveles Natural, Low, Mid, High.")
    else:
        st.write("Calcula promedios y desviaciones para todos los archivos agrupados por nivel de dificultad.")
        if st.button("🚀 Ejecutar Análisis Comparativo Global"):
            with st.spinner("Procesando archivos de todos los niveles..."):
                active_indices = [ch_index(name) for name in visible_channels if ch_index(name) is not None]
                cog_results = {}
                rel_bands_results = {}
                bm_results = {lvl: {"theta_fz": [], "alpha": [], "faa": [], "burst_rate": []} for lvl in levels_dict}

                for lvl, files_list in levels_dict.items():
                    cog_vals = {}
                    rb_vals = {b: [] for b in BANDS}
                    
                    for fn in files_list:
                        fp = os.path.join(dataset_dir, fn)
                        try:
                            x_clean = load_and_preprocess_file(fp, fs, bandpass, notch_val, trim_seconds=trim_seconds)
                            dur_f = len(x_clean) / fs
                            if dur_f < 4.0:
                                continue
                            
                            # PSD de canales visibles
                            f_c, P_c = compute_psd(x_clean[:, active_indices], fs)
                            bp_c = band_powers(f_c, P_c)
                            tot_c = sum(bp_c.values())
                            
                            for k, v in cognitive_indices(bp_c).items():
                                cog_vals.setdefault(k, []).append(float(np.mean(v)))
                            for b in BANDS:
                                rb_vals[b].append(float(np.mean(bp_c[b] / tot_c)) * 100)

                            # Biomarcadores
                            _, _, _, f_bursts = detect_theta_bursts(x_clean[:, th_idx], fs, ratio=burst_ratio)
                            s_f = biomarker_summary(x_clean, fs, f_bursts, dur_f)
                            for k_bm in ["theta_fz", "alpha", "faa", "burst_rate"]:
                                bm_results[lvl][k_bm].append(s_f[k_bm])
                        except Exception:
                            continue
                            
                    cog_results[lvl] = cog_vals
                    rel_bands_results[lvl] = rb_vals

                # Renderizar Gráficas de Bandas e Índices
                st.markdown("### 1. Índices Cognitivos y Potencia Relativa por Nivel")
                fig_cmp1 = plt.figure(figsize=(12, 6))
                levels_keys = list(levels_dict.keys())
                keys_idx = ["Theta/Alpha", "Beta/Alpha", "Engagement β/(α+θ)"]
                
                for j, k_name in enumerate(keys_idx):
                    ax = fig_cmp1.add_subplot(2, 3, j + 1)
                    mu = [np.mean(cog_results[l].get(k_name, [np.nan])) for l in levels_keys]
                    sd = [np.std(cog_results[l].get(k_name, [np.nan])) for l in levels_keys]
                    ax.bar(levels_keys, mu, yerr=sd, capsize=4, color="#4c78a8", alpha=0.85)
                    ax.set_title(k_name, fontsize=10)
                    ax.tick_params(axis="x", labelsize=8, rotation=15)
                    ax.grid(True, axis="y", ls="--", alpha=0.4)

                ax_b = fig_cmp1.add_subplot(2, 1, 2)
                w = 0.8 / len(BANDS)
                x0 = np.arange(len(levels_keys))
                for i_b, b_name in enumerate(BANDS):
                    m_vals = [np.mean(rel_bands_results[l][b_name]) if rel_bands_results[l][b_name] else np.nan for l in levels_keys]
                    ax_b.bar(x0 + i_b * w, m_vals, w, label=b_name, color=BAND_COLORS[b_name])
                ax_b.set_xticks(x0 + 0.4 - w / 2)
                ax_b.set_xticklabels(levels_keys)
                ax_b.set_ylabel("Potencia relativa (%)")
                ax_b.set_title("Potencia relativa por banda y nivel (Post-Trim 3s)")
                ax_b.legend(ncol=4, fontsize=8)
                ax_b.grid(True, axis="y", ls="--", alpha=0.4)
                fig_cmp1.tight_layout()
                st.pyplot(fig_cmp1)
                plt.close(fig_cmp1)

                # Renderizar Gráfica de Biomarcadores
                st.markdown("### 2. Biomarcadores Específicos por Nivel")
                _, a_lbl = alpha_channels()
                titles_bm = {
                    "theta_fz": "Theta tónico Fz (µV²)",
                    "alpha": f"Alfa tónico {a_lbl} (µV²)",
                    "faa": "FAA = ln α(F4) − ln α(F3)",
                    "burst_rate": "Ráfagas theta fásico (por min)"
                }
                fig_cmp2 = plt.figure(figsize=(12, 6))
                colors_lvl = ["#7f7f7f", "#2ca02c", "#ff7f0e", "#d62728"]
                
                for j, k_key in enumerate(["theta_fz", "alpha", "faa", "burst_rate"]):
                    ax = fig_cmp2.add_subplot(2, 2, j + 1)
                    for i_l, l_name in enumerate(levels_keys):
                        arr = np.array([z for z in bm_results[l_name][k_key] if np.isfinite(z)])
                        if len(arr) == 0:
                            continue
                        ax.bar(i_l, arr.mean(), yerr=arr.std() if len(arr) > 1 else 0, capsize=4,
                               color=colors_lvl[i_l % len(colors_lvl)], alpha=0.55)
                        ax.scatter(np.full(len(arr), i_l) + np.linspace(-0.12, 0.12, len(arr)), arr,
                                   color="k", s=14, zorder=3)
                    ax.set_xticks(range(len(levels_keys)))
                    ax.set_xticklabels(levels_keys, fontsize=8)
                    ax.set_title(titles_bm[k_key], fontsize=10)
                    ax.grid(True, axis="y", ls="--", alpha=0.4)
                    if k_key == "faa":
                        ax.axhline(0, color="gray", lw=1)
                fig_cmp2.tight_layout()
                st.pyplot(fig_cmp2)
                plt.close(fig_cmp2)
                st.success("✅ Comparación completada exitosamente.")
