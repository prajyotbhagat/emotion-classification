import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
from model import VGG16LSTM
import io
import json
import base64
import streamlit.components.v1 as components
import plotly.graph_objs as go




st.set_page_config(page_title="🐬 Delle", layout="centered")

idx2label = {
    0: "angry", 1: "calm", 2: "disgust", 3: "fearful",
    4: "happy", 5: "neutral", 6: "sad", 7: "surprised"
}

@st.cache_resource
def load_model():
    model = VGG16LSTM(num_classes=8, fine_tune=False)
    model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

def preprocess_audio_mic(audio, sr=16000, n_mels=128, max_len=3):
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  
    y = librosa.util.fix_length(audio, int(sr * max_len))
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    melspec_db = (melspec_db - melspec_db.mean()) / (melspec_db.std() + 1e-6)
    tensor = torch.tensor(melspec_db).unsqueeze(0).unsqueeze(0)
    return tensor.float()

def preprocess_audio(audio_bytes, sr=16000, n_mels=128, max_len=3):
    import soundfile as sf

    
    y, _ = sf.read(io.BytesIO(audio_bytes), dtype='float32')

    
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    
    y = librosa.util.fix_length(y, size=int(sr * max_len))

    
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    melspec_db = (melspec_db - melspec_db.mean()) / (melspec_db.std() + 1e-6)

    tensor = torch.tensor(melspec_db).unsqueeze(0).unsqueeze(0)
    return tensor.float()

def plot_melspectrogram(melspec_db, sr=16000, hop_length=512):
    st.markdown(
        "<h2 style='text-align: center; color: white;'>🎼 Mel-Spectrogram</h2>",
        unsafe_allow_html=True
    )

    num_mels, num_frames = melspec_db.shape
    time_axis = np.arange(num_frames) * hop_length / sr
    mel_axis = np.linspace(0, sr // 2, num_mels)

    fig = go.Figure(data=go.Heatmap(
        z=melspec_db,
        x=time_axis,
        y=mel_axis,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text='dB',
                side='right',
                font=dict(color='white')
            ),
            tickfont=dict(color='white'),
            outlinecolor='white'
        )
    ))

    fig.update_layout(
        xaxis=dict(title='Time (s)', color='white', showgrid=False),
        yaxis=dict(title='Mel Frequency', color='white', showgrid=False),
        paper_bgcolor='#0f1117',
        plot_bgcolor='#0f1117',
        margin=dict(t=30, r=20, l=60, b=60)
    )

    st.plotly_chart(fig, use_container_width=True)



def show_animated_waveform(y, sr=16000, max_points=5000):

    
    times = np.arange(len(y)) / sr

    # Downsample if necessary
    if len(y) > max_points:
        factor = len(y) // max_points
        y_ds = y[::factor]
        times_ds = times[::factor]
    else:
        y_ds = y
        times_ds = times

    waveform_data = {
        "x": times_ds.tolist(),
        "y": y_ds.tolist()
    }

    global base64_audio

    html = f"""
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body style="background-color:#0f1117;">
        <h1 style="color:white;text-align:center;">🎵 Live Audio Waveform</h1>
        <audio id="audio" controls style="width:100%;margin-bottom:10px;">
            <source src="data:audio/wav;base64,{base64_audio}" type="audio/wav">
        </audio>
        <div id="plot" style="width:100%;height:350px;"></div>

        <script>
            const x_vals = {json.dumps(waveform_data["x"])};
            const y_vals = {json.dumps(waveform_data["y"])};
            const duration = x_vals[x_vals.length - 1];
            const view_window = 6.0;
            const audio = document.getElementById("audio");

            const layout = {{
                paper_bgcolor: '#0f1117',
                plot_bgcolor: '#0f1117',
                xaxis: {{
                    title: 'Time (s)',
                    color: 'white',
                    showgrid: false,
                    tickformat: '.2f',
                    tickfont: {{ color: 'white' }},
                    hoverformat: '.2f'
                }},
                yaxis: {{
                    title: 'Amplitude',
                    color: 'white',
                    showgrid: false,
                    tickformat: '.1f',
                    tickfont: {{ color: 'white' }},
                    hoverformat: '.2f'
                }},
                showlegend: false,
                margin: {{ t: 40, r: 30, l: 40, b: 40 }}
            }};

            Plotly.newPlot("plot", [], layout, {{responsive: true}});
            let animId;

            function interpolateAmplitude(time) {{
                // If outside the data range, return 0
                if (time <= x_vals[0] || time >= x_vals[x_vals.length - 1]) return 0;

                for (let i = 1; i < x_vals.length; i++) {{
                    if (x_vals[i] >= time) {{
                        const t0 = x_vals[i - 1];
                        const t1 = x_vals[i];
                        const y0 = y_vals[i - 1];
                        const y1 = y_vals[i];
                        const ratio = (time - t0) / (t1 - t0);
                        return y0 + ratio * (y1 - y0);
                    }}
                }}
                return 0;
            }}

            function animate() {{
                const time = audio.currentTime;
                const markerY = interpolateAmplitude(time);

                const left = Math.max(0, time - view_window);
                const right = time;

                const windowX = [];
                const windowY = [];

                for (let i = 0; i < x_vals.length; i++) {{
                    if (x_vals[i] >= left && x_vals[i] <= right) {{
                        windowX.push(x_vals[i]);
                        windowY.push(y_vals[i]);
                    }}
                }}

                const baseTrace = {{
                    x: windowX,
                    y: windowY,
                    mode: 'lines',
                    line: {{ color: 'rgba(0,255,200,0.9)', width: 2 }},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0,255,200,0.2)'
                }};

                const markerTrace = {{
                    x: [time],
                    y: [markerY],
                    mode: 'markers',
                    marker: {{ color: 'red', size: 10 }},
                    showlegend: false
                }};

                Plotly.react("plot", [baseTrace, markerTrace], layout);
                Plotly.relayout("plot", {{
                    "xaxis.range": [left, right]
                }});

                animId = requestAnimationFrame(animate);
            }}

            audio.onplay = function() {{
                setTimeout(() => {{
                    cancelAnimationFrame(animId);
                    animate();
                }}, 150);
            }}

            audio.onpause = function() {{
                cancelAnimationFrame(animId);
            }}

            audio.onended = function() {{
                cancelAnimationFrame(animId);
            }}
        </script>
    </body>
    </html>
    """

    components.html(html, height=450)



model = load_model()




show_beat_graph = st.sidebar.checkbox("🎵 Show Live Waveform", value=False)
show_spectrogram = st.sidebar.checkbox("🎛️ Show Mel-Spectrogram", value=False)


st.markdown(
    "<h2 style='text-align: center; color: white;'>🐬 Delle</h2>",
    unsafe_allow_html=True
)
st.subheader("Upload an Audio File for Emotion Detection")

uploaded_file = st.file_uploader("", type=["mp3", "wav"])




if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    st.audio(file_bytes, format='audio/mp3')  

    y, _ = sf.read(io.BytesIO(file_bytes), dtype='float32')
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    tensor_input = preprocess_audio(file_bytes)

    if tensor_input is not None:
        with torch.no_grad():
            output = model(tensor_input)
            pred_class = torch.argmax(output, dim=1).item()
            emotion = idx2label[pred_class]

        
        st.success("✅ Audio processed successfully!")
        st.markdown(f"### 🎭 Emotion: **{emotion.upper()}**")

        
        emotion_icons = {
            "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
            "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲"
        }
        st.markdown(f"<h1 style='text-align: center'>{emotion_icons[emotion]}</h1>", unsafe_allow_html=True)
        
        
    
    if show_beat_graph:
        base64_audio = base64.b64encode(file_bytes).decode("utf-8")
        show_animated_waveform(y, sr=16000)

       
    if show_spectrogram:
        melspec = tensor_input.squeeze().numpy()
        plot_melspectrogram(melspec)

st.markdown("</div>", unsafe_allow_html=True)

st.caption("✨ Powered by VGG16 + LSTM on Mel-Spectrograms of speech.")



