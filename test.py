import torch
import numpy as np
import librosa
import argparse
import os
from model import VGG16LSTM

# Constants
MODEL_PATH = "emotion_model.pth"
SR = 16000
N_MELS = 128
MAX_LEN = 3

# Label mapping
idx2label = {
    0: "angry", 1: "calm", 2: "disgust", 3: "fearful",
    4: "happy", 5: "neutral", 6: "sad", 7: "surprised"
}

def load_model():
    model = VGG16LSTM(num_classes=8, fine_tune=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_audio(path, sr=SR, n_mels=N_MELS, max_len=MAX_LEN):
    y, _ = librosa.load(path, sr=sr)
    target_len = sr * max_len
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    
    melspec_db = (melspec_db - melspec_db.mean()) / (melspec_db.std() + 1e-6)
    return torch.tensor(melspec_db).unsqueeze(0).unsqueeze(0).float()

def predict(audio_path):
    model = load_model()
    tensor_input = preprocess_audio(audio_path)

    with torch.no_grad():
        output = model(tensor_input)
        pred_idx = torch.argmax(output, dim=1).item()
        emotion = idx2label[pred_idx]

    return emotion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion prediction from audio file.")
    parser.add_argument("--file", type=str, required=True, help="Path to the audio file (.wav or .mp3)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        exit(1)

    emotion = predict(args.file)
    print(f"🎧 Predicted Emotion: {emotion.upper()}")

