import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque
from PIL import Image
import time

# --- Model Definition ---
class VideoMambaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

# --- Load Model ---
@st.cache_resource
def load_model():
    model = VideoMambaModel()
    model.load_state_dict(torch.load("videomamba_anomaly.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- Parameters ---
CLIP_LENGTH = 10
THRESHOLD = 0.5
clip_buffer = deque(maxlen=CLIP_LENGTH)

st.set_page_config(page_title="VideoMamba Anomaly Detection", layout="wide")
st.title("🎥 VideoMamba - Live Anomaly Detection")

col1, col2 = st.columns([3, 1])
FRAME_WINDOW = col1.image([])
status_placeholder = col2.empty()

start = st.button("▶️ Start Live Monitoring")
stop = st.button("⏹ Stop")

if start:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Could not access camera. Please allow webcam permission.")
    else:
        st.success("✅ Webcam started. Monitoring anomalies...")
        time.sleep(1)

        while cap.isOpened():
            if stop:
                st.warning("🛑 Monitoring stopped.")
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from webcam.")
                break

            # Preprocess frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            normalized = resized.astype(np.float32) / 255.0
            clip_buffer.append(normalized)

            label = "Collecting frames..."
            if len(clip_buffer) == CLIP_LENGTH:
                clip_tensor = np.stack(clip_buffer)[np.newaxis, np.newaxis, ...]
                clip_tensor = torch.tensor(clip_tensor, dtype=torch.float32)
                with torch.no_grad():
                    score = model(clip_tensor).item()
                label = "🚨 **Anomaly Detected!**" if score > THRESHOLD else "✅ Normal"
                label += f"  _(Score: {score:.3f})_"

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            status_placeholder.markdown(f"### {label}")

        cap.release()
