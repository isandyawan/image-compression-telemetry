import streamlit as st
from PIL import Image
import io
import time
import base64
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.autoencoder import AutoEncoder

@st.cache_resource
def load_autoencoder():
    return AutoEncoder("Backend/final_model")

model = load_autoencoder()

st.set_page_config(page_title="End-to-end Satellite Image Compression Autoencoder", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Space Mono', monospace;
        color: #e0eaf5;
    }
    
    /* Styling Headers */
    h1, h2, h3 {
        color: #8ea2a3 !important; 
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Styling Buttons */
    .stButton>button, .stDownloadButton>button {
        background-color: transparent !important;
        border: 1px solid #8ea2a3 !important;
        color: #8ea2a3 !important;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #8ea2a3 !important;
        color: #050b14 !important;
        box-shadow: 0 0 10px #8ea2a3;
    }
    
    /* Panel Boxes */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        background-color: rgba(10, 25, 47, 0.7);
        border: 1px solid #1e3d59;
        border-radius: 8px;
        padding: 20px;
    }
    
    /* Image caption */
    .st-emotion-cache-1kyxreq {
        color: #a8b2d1;
        font-size: 0.8rem;
        text-transform: uppercase;
    }

    /* Styling khusus untuk st.metric */
    div[data-testid="stMetricValue"] {
        color: #00ff00 !important; 
    }
    div[data-testid="stMetricLabel"] {
        color: #8ea2a3 !important; 
    }
    </style>
""", unsafe_allow_html=True)


def add_video_background(video_file_path):
    with open(video_file_path, "rb") as video_file:
        video_encoded = base64.b64encode(video_file.read()).decode()
    
    st.markdown(f"""
        <style>
        #myVideo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100vw; 
            min-height: 100vh;
            z-index: -1; 
            object-fit: cover;
            filter: brightness(0.4); 
        }}
        .stApp {{
            background: transparent !important;
        }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}
        </style>
        
        <video autoplay muted loop playsinline id="myVideo">
            <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4">
        </video>
        """, unsafe_allow_html=True)
    
try:
    add_video_background("frontend/bg.mp4")
except FileNotFoundError:
    st.warning("Video background tidak ditemukan. Pastikan file bg.mp4 ada di folder ini.")

st.title("End-to-end Satellite Image Compression Autoencoder")
st.markdown("---")

# Layout 3 Kolom: [Upload] | [Proses & Metrik] | [Hasil]
col1, col2, col3 = st.columns([1, 1.2, 1])

# ==========================================
# KOLOM 1: UPLOAD & RAW DATA
# ==========================================
with col1:
    st.subheader("1.⁠ IMAGE INGESTION")
    uploaded_file = st.file_uploader("UPLOAD RAW IMAGE", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # with open("/tmp/temp.png", "wb") as f:
        #     f.write(uploaded_file.getbuffer())
        
        uploaded_bytes = uploaded_file.getvalue()
        file_id = uploaded_file.name + str(len(uploaded_bytes))
        image = Image.open(io.BytesIO(uploaded_bytes)).convert('RGB')
        raw_size_bytes = len(uploaded_bytes)

        st.session_state.raw_size_kb = raw_size_bytes / 1024
        
        st.image(image, caption="RAW OPTICAL DATA", use_container_width=True)        
    else:
        st.info("AWAITING INGESTION...")

# ==========================================
# KOLOM 2: PROCESSING & METRICS
# ==========================================
with col2:
    st.subheader("2.⁠ LATENT IMAGE DETAILS")
    
    if uploaded_file is None:
        st.info("SYSTEM STANDBY...")
    else:
        status_text = st.empty()
        progress_bar = st.progress(100)

        if st.session_state.get("file_id") != file_id:
            st.session_state.file_id = file_id
            progress_bar.progress(0)
            
            status_text.markdown(f"**STATUS:** `UPLOADING IMAGE`")
            progress_bar.progress(5)
            time.sleep(0.4)

            status_text.markdown(f"**STATUS:** `COMPRESSING IMAGE`")
            progress_bar.progress(6)
            st.session_state.buffer_img = model.compress_tensor(uploaded_bytes)

            status_text.markdown(f"**STATUS:** `COMPRESSION COMPLETED`")
            progress_bar.progress(50)
            st.session_state.buffer_img_io = io.BytesIO(st.session_state.buffer_img)
            
            status_text.markdown(f"**STATUS:** `CALCULATING METRICS`")
            progress_bar.progress(60)
            payload_size_bytes = len(st.session_state.buffer_img_io.getvalue())
            st.session_state.payload_size_kb = payload_size_bytes / 1024
            st.session_state.compression_ratio = (1 - (payload_size_bytes / raw_size_bytes)) * 100
            st.markdown("---")

            status_text.markdown(f"**STATUS:** `IMAGE DECOMPRESSION IN PROGRESS`")
            progress_bar.progress(70)
            st.session_state.reconstructed_img = model.decompress(st.session_state.buffer_img)  # Simulate decompression untuk generate reconstructed_img
                 
        buffer_img = st.session_state.buffer_img
        buffer_img_io = st.session_state.buffer_img_io
        reconstructed_img = st.session_state.reconstructed_img
        raw_size_kb = st.session_state.raw_size_kb
        payload_size_kb = st.session_state.payload_size_kb
        compression_ratio = st.session_state.compression_ratio

        st.metric(label="RAW IMAGE SIZE", value=f"{raw_size_kb:.1f} KB")
        st.metric(label="LATENT PAYLOAD", value=f"{payload_size_kb:.1f} KB")
        st.metric(label="COMPRESSION RATIO", value=f"{compression_ratio:.1f}%")
            
        st.markdown("---")        
        status_text.markdown(f"**STATUS:** `CALCULATING METRICS`")
        progress_bar.progress(95)                    

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="DL LATENT (.npy)",
                data=buffer_img_io.getvalue(),
                file_name="/tmp/telemetry_latent.npy",
                mime="application/octet-stream",
                use_container_width=True
            )
            
        with dl_col2:
            st.download_button(
                label="DL DECODED (.png)",
                data=reconstructed_img,
                file_name="/tmp/decoded_telemetry.png",
                mime="image/png",
                use_container_width=True
            )

        fidelity_score = model.calculate_metrics_from_bytes(uploaded_bytes, reconstructed_img)
        status_text.markdown(f"**STATUS:** `PROCESSING COMPLETED`")
        progress_bar.progress(100)
        # Grid Metrik 2x2 biar muat di kolom tengah

        
# ==========================================
# KOLOM 3: RECONSTRUCTED PREVIEW
# ==========================================
with col3:
    st.subheader("3.⁠ DECODED IMAGE")
    
    if uploaded_file is None:
        st.info("AWAITING PAYLOAD...")
    else:
        st.caption("Click to expand")
        st.image(reconstructed_img, caption="RECONSTRUCTED TELEMETRY", use_container_width=True)
        st.metric(label="FIDELITY", value=f"{fidelity_score:.2f}%")
