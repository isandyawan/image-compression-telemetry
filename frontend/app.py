import streamlit as st
# import torch.nn as nn
from PIL import Image
import io
import time
import base64
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.autoencoder import AutoEncoder

# --- 1. DEFINISI MODEL AUTOENCODER ---
# class SimpleAutoencoder(nn.Module):
#     def __init__(self):
#         super(SimpleAutoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 16, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         latent = self.encoder(x)
#         reconstructed = self.decoder(latent)
#         return reconstructed, latent

# @st.cache_resource
# def load_model():
#     model = SimpleAutoencoder()
#     try:
#         model.load_state_dict(torch.load('autoencoder_weights.pth', weights_only=True, map_location=torch.device('cpu')))
#     except FileNotFoundError:
#         pass # Akan jalan dengan untrained model jika file tidak ada
#     model.eval()
#     return model

@st.cache_resource
def load_autoencoder():
    return AutoEncoder("Backend/final_model")

model = load_autoencoder()

# --- 2. SETUP UI & CUSTOM CSS (NASA THEME) ---
st.set_page_config(page_title="NASA | Orbital Compression", layout="wide", initial_sidebar_state="collapsed")

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

# --- 3. APLIKASI UTAMA ---
st.title("IMAGE COMPRESSION FOR SPACE TELEMETRY")
st.markdown("---")

# Layout 3 Kolom: [Upload] | [Proses & Metrik] | [Hasil]
col1, col2, col3 = st.columns([1, 1.2, 1])

# ==========================================
# KOLOM 1: UPLOAD & RAW DATA
# ==========================================
with col1:
    st.subheader("1. TELEMETRY INGESTION")
    uploaded_file = st.file_uploader("UPLOAD RAW IMAGE", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        with open("/tmp/temp.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_bytes = uploaded_file.read()
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="RAW OPTICAL DATA", use_container_width=True)

        raw_size_bytes = len(uploaded_file.getvalue())
        raw_size_kb = raw_size_bytes / 1024
    else:
        st.info("AWAITING INGESTION...")

# ==========================================
# KOLOM 2: PROCESSING & METRICS
# ==========================================
with col2:
    st.subheader("2. SYSTEM & METRICS")
    
    if uploaded_file is None:
        st.info("SYSTEM STANDBY...")
    else:
        # transform = transforms.Compose([
        #     transforms.Resize((128, 128)), 
        #     transforms.ToTensor()
        # ])
        # input_tensor = transform(image).unsqueeze(0)

        status_text = st.empty()
        progress_bar = st.progress(0)
        
        steps = [
            (20, "ANALYZING DENSITY..."),
            (50, "QUANTIZING VECTORS..."),
            (80, "CALCULATING METRICS..."),
            (100, "RECONSTRUCTION VERIFIED.")
        ]
        
        for percent, text in steps:
            status_text.markdown(f"**STATUS:** `{text}`")
            progress_bar.progress(percent)
            time.sleep(0.4) 
            
        status_text.markdown("**STATUS:** `<span style='color:#00ff00'>SYSTEM NOMINAL.</span>`", unsafe_allow_html=True)

        # with torch.no_grad():
        #     reconstructed_tensor, latent_tensor = model(input_tensor)

        # reconstructed_img_tensor = reconstructed_tensor.squeeze(0)
        # reconstructed_img = transforms.ToPILImage()(reconstructed_img_tensor)
        # latent_array = latent_tensor.squeeze(0).numpy()

        # buffer_latent = io.BytesIO()
        # np.save(buffer_latent, latent_array)
        # buffer_img = io.BytesIO()
        # buffer_img = model.compress(image)
        # buffer_img = model.compress("temp.png")
        buffer_img = model.compress_tensor(uploaded_bytes)
        buffer_img_io = io.BytesIO(buffer_img)
        
        # reconstructed_img.save(buffer_img, format="PNG")

        payload_size_bytes = len(buffer_img_io.getvalue())
        payload_size_kb = payload_size_bytes / 1024
        compression_ratio = (1 - (payload_size_bytes / raw_size_bytes)) * 100
        # mse_loss = torch.mean((input_tensor - reconstructed_tensor) ** 2).item()
        # fidelity_score = max(0.0, (1.0 - mse_loss) * 100) 
        ##TODO workaround for fidelity score 
        # fidelity_score = 85.0
        
        st.markdown("---")
        
        reconstructed_img = model.decompress(buffer_img)  # Simulate decompression untuk generate reconstructed_img

        fidelity_score = model.calculate_metrics_from_bytes(uploaded_bytes, reconstructed_img)
        
        # Grid Metrik 2x2 biar muat di kolom tengah
        m1, m2 = st.columns(2)
        m1.metric(label="RAW SIZE", value=f"{raw_size_kb:.1f} KB")
        m2.metric(label="PAYLOAD", value=f"{payload_size_kb:.1f} KB")
        
        m3, m4 = st.columns(2)
        m3.metric(label="COMPRESSED", value=f"{compression_ratio:.1f}%")
        m4.metric(label="FIDELITY", value=f"{fidelity_score:.2f}%")

        st.markdown("---")
        
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

# ==========================================
# KOLOM 3: RECONSTRUCTED PREVIEW
# ==========================================
with col3:
    st.subheader("3. DECODED OUTPUT")
    
    if uploaded_file is None:
        st.info("AWAITING PAYLOAD...")
    else:
        st.caption("Click to expand")
        st.image(reconstructed_img, caption="RECONSTRUCTED TELEMETRY", use_container_width=True)