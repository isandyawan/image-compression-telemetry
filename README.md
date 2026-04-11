# End-to-end Satellite Image Compression Autoencoder

A high-performance satellite imagery compression dashboard built with **Streamlit** and **Deep Learning**. This system utilizes a Convolutional Autoencoder to compress raw optical data into latent space vectors for efficient telemetry transmission.

-----

## 🚀 Features

  * **Telemetry Ingestion:** Upload raw satellite imagery (JPG/PNG).
  * **Real-time Processing:** Live encoding and decoding status updates.
  * **Performance Metrics:** Automatic calculation of Raw Size, Payload Size, Compression Ratio, and Fidelity Score.
  * **Latent Space Export:** Download the compressed .npy latent representation or the reconstructed .png.

-----

##  Local Setup (Docker)
### 1\. Clone Repository

git clone https://huggingface.co/spaces/isandyawan/compression-telemetry
cd compression-telemetry

### 2\. Build Docker Image

docker build -t satellite-compression:latest .

### 3\. Run Container

docker run -d -p 8501:8501 --name satellite-app satellite-compression:latest

### 4\. Access the Dashboard

Open browser and navigate to http://localhost:8501

-----

## Project Structure

  * app.py - Streamlit frontend with customised Cyberpunk UI.
  * Backend/ - Autoencoder logic.
  * frontend/ - Visual assets and homepage.
  * Dockerfile - Docker environment configuration.

-----

## Core Technologies

  * **Frontend:** Streamlit
  * **Deep Learning:** TensorFlow / Keras
  * **Containerization:** Docker

-----

## Acknowledgements
This project built using open-source model and following resources:
  * **Model Architecture:** Developed based on research by Mahmoud Ashraf on AutoencoderCompression (https://github.com/MahmoudAshraf97/AutoencoderCompression).
  * **AI Assistance:** Google Gemini for UI optimisation and assets generator. 

-----

## License

Distributed under the MIT License.