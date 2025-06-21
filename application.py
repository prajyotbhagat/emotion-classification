import streamlit as st
import torch
import numpy as np
import librosa
import io
from model import VGG16LSTM

st.set_page_config(page_title="Speech Emotion Classifier", page_icon="🎧", layout="centered")


st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(to right, #e0f7fa, #e1bee7);
            padding: 2rem;
            border-radius: 10px;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #4a148c;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🎧 Real-Time Emotion Classifier</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = VGG16LSTM(num_classes=8, fine_tune=False)
    model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_audio(audio_bytes, sr=16000, n_mels=128, max_len=3):
    y, _ = librosa.load(audio_bytes, sr=sr)
    y = librosa.util.fix_length(y, size=int(sr * max_len))
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    melspec_db = (melspec_db - melspec_db.mean()) / (melspec_db.std() + 1e-6)
    tensor = torch.tensor(melspec_db).unsqueeze(0).unsqueeze(0)
    return tensor.float()

idx2label = {
    0: "angry", 1: "calm", 2: "disgust", 3: "fearful",
    4: "happy", 5: "neutral", 6: "sad", 7: "surprised"
}

uploaded_file = st.file_uploader("🎙️ Upload an Audio File (.mp3)", type=["mp3"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    st.audio(file_bytes, format='audio/mp3')

    model = load_model()
    tensor_input = preprocess_audio(io.BytesIO(file_bytes))

    if tensor_input is not None:
        with torch.no_grad():
            output = model(tensor_input)
            pred_class = torch.argmax(output, dim=1).item()
            emotion = idx2label[pred_class]

        
        st.success("✅ Audio processed successfully!")
        st.markdown(f"### 🎭 Predicted Emotion: **{emotion.upper()}**")

        
        emotion_icons = {
            "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
            "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲"
        }
        st.markdown(f"<h1 style='text-align: center'>{emotion_icons[emotion]}</h1>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.caption("✨ Powered by VGG16 + LSTM on Mel-Spectrograms of speech.")



