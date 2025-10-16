import io
import torch
import uvicorn
import torch.nn as nn
import soundfile
import torch.nn.functional as F
from torchaudio import transforms
from fastapi import FastAPI, HTTPException, UploadFile, File


class CheckAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, audio):
        audio = audio.unsqueeze(1)
        audio = self.first(audio)
        audio = self.second(audio)
        return audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.MelSpectrogram(
     sample_rate = 22050,
     n_mels = 64
 )
max_len = 500
genres = torch.load('label.pth')
# index_to_label = {label:ind for ind, label in enumerate(genres)}
index_to_label = {ind: label for ind, label in enumerate(genres)}

model = CheckAudio()
model.load_state_dict(torch.load('audio_model.pth', map_location=device))
model.to(device)

def change_audio(waveform, sample_rate):
    if sample_rate != 22050:
        resample = transforms.Resample(orig_freq=sample_rate, new_freq=22050)
        # waveform = resample(torch.tensor(waveform).unsqueeze(0))
        waveform = resample(waveform.unsqueeze(0))
    input_spectrogram = transform(waveform).squeeze(0)
    if input_spectrogram.shape[1] > max_len:
        input_spectrogram = input_spectrogram[:, :max_len]
    elif input_spectrogram.shape[1] < max_len:
        pad_len = max_len - input_spectrogram.shape[1]
        input_spectrogram = F.pad(input_spectrogram, (0, pad_len))
    return input_spectrogram


app = FastAPI()

@app.post('/predict/')
async def predict_audio(file:UploadFile=File(...)):
    try:
        audio = await file.read()
        if not audio:
            raise HTTPException(status_code=400, detail='It is error file')
        waveform, sample_rate = soundfile.read(io.BytesIO(audio), dtype='float32')
        # waveform = torch.tensor(waveform).T
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if waveform.ndim > 1:
            waveform = waveform.flatten()

        spec = change_audio(waveform, sample_rate).unsqueeze(0).to(device)

        with torch.no_grad():
            y_prediction = model(spec)
            prediction_int = torch.argmax(y_prediction, dim=1).item()
            prediction_class = index_to_label[prediction_int]
        return {"result": f"Index of class: {prediction_int}, Class name(genre): {prediction_class}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
