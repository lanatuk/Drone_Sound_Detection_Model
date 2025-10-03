import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gradio as gr
from model import AudioEfficientNet
import librosa

# Load model
model = AudioEfficientNet(num_classes=2)
model.load_state_dict(torch.load("trained_model_weights.pth", map_location="cpu"))
model.eval()

target_sr = 16000
target_seconds = 1.0
target_length = int(target_sr * target_seconds)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=target_sr,
    n_mels=128,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    f_min=0.0,
    f_max=target_sr / 2,
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

def predict(audio):
    # Load audio 
    waveform, sr = librosa.load(audio, sr=target_sr, mono=True)   
    waveform = torch.tensor(waveform, dtype=torch.float32)        

    if waveform.abs().max() > 1.0:
        waveform = waveform / waveform.abs().max()

    # Pad/Crop
    if waveform.numel() > target_length:
        start = int((waveform.numel() - target_length) // 2)        
        waveform = waveform[start:start + target_length]
    else:
        padding = int(target_length - waveform.numel())
        waveform = F.pad(waveform, (0, padding))

    # Mel spectrogram 
    mel = mel_transform(waveform.unsqueeze(0))
    mel_db = amplitude_to_db(mel)

    # Normalization
    mean = mel_db.mean()
    std = mel_db.std().clamp(min=1e-6)
    mel_db = (mel_db - mean) / std

    # Resize 
    mel_resized = F.interpolate(
        mel_db.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False
    ).squeeze(0)
    image_3ch = mel_resized.repeat(3,1,1).unsqueeze(0)  # add batch dimension

    # Prediction
    with torch.no_grad():
        outputs = model(image_3ch)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    labels = ["No Drone", "Drone"]

    # Scale spectrogram for display
    mel_vis = mel_db.squeeze().numpy()
    mel_vis = (mel_vis - mel_vis.min()) / (mel_vis.max() - mel_vis.min() + 1e-6)  # norm 0-1
    mel_vis = (mel_vis * 255).astype("uint8")  # 0-255 uint8

    return {labels[i]: float(probs[i]) for i in range(len(labels))}, mel_vis

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", format="wav"),
    outputs=[gr.Label(num_top_classes=2), gr.Image(type="numpy")],
    title="Drone Detection Demo",
    description="Upload or record audio. The model predicts if a drone is present and shows the spectrogram."
)

demo.launch()