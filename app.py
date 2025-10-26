import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque

st.set_page_config(page_title="VideoMamba Live Anomaly Detection")

# 3D CNN model
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
        self.fc = nn.Linear(16,1)
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

# Load model
model = VideoMambaModel()
model.load_state_dict(torch.load("videomamba_anomaly.pth", map_location="cpu"))
model.eval()

CLIP_LENGTH = 10
THRESHOLD = 0.5
clip_buffer = deque(maxlen=CLIP_LENGTH)

# Video transformer
class AnomalyDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128,128))
        normalized = resized.astype(np.float32)/255.0
        clip_buffer.append(normalized)

        label = "Collecting frames..."
        if len(clip_buffer) == CLIP_LENGTH:
            clip_tensor = np.stack(clip_buffer)[np.newaxis, np.newaxis,...]
            clip_tensor = torch.tensor(clip_tensor, dtype=torch.float32)
            with torch.no_grad():
                score = model(clip_tensor).item()
            label = "🚨 Anomaly" if score > THRESHOLD else "✅ Normal"
            label += f" ({score:.3f})"

        # Overlay text on frame
        cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return img

st.title("🚨 VideoMamba Live Anomaly Detection")
st.write("Live webcam feed with real-time anomaly detection.")

webrtc_streamer(
    key="video_mamba",
    video_transformer_factory=AnomalyDetector,
    media_stream_constraints={"video": True, "audio": False}
)
