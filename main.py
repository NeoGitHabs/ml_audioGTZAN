from fastapi import FastAPI, HTTPException, UploadFile, File
from torchaudio import transforms as T
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torch.nn as nn
import soundfile as sf
import numpy as np
import torchaudio
import tempfile
import librosa
import uvicorn
import torch
import io
import os



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


classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
index_to_label = {label: ind for ind, label in enumerate(classes)}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CheckAudio()
model.load_state_dict(torch.load('audioGTZAN.pth', map_location=device))
model.to(device)
model.eval()

transform = T.MelSpectrogram(
     sample_rate = 22050,
     n_mels = 64
)

max_len = 500

def change_audio(waveform, sr):
    if sr != 22050:
        resample = T.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)

    input_spectrogram = transform(waveform).squeeze(0)

    if input_spectrogram.shape[1] > max_len:
        input_spectrogram = input_spectrogram[:, :max_len]
    elif input_spectrogram.shape[1] < max_len:
        pad_len = max_len - input_spectrogram.shape[1]
        input_spectrogram = F.pad(input_spectrogram, (0, pad_len))

    return input_spectrogram


app = FastAPI()

@app.post('/predict')
async def predict_audio(file:UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Пустой файл')

        wf, sr = sf.read(io.BytesIO(data), dtype='float')
        wf = torch.tensor(wf)

        spec = change_audio(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_label[pred_ind]
            return {f'Индекс: {pred_ind}, Жанр: {pred_class}'}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{e}')

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


# st.title('Audio Genre Classifier')
# st.text('Загрузите аудио, и модель попробует её распознать.')
#
# mnist_audio = st.file_uploader('Выберите аудио', type=['wav', 'mp3', 'flac', 'ogg'])
#
# if not mnist_audio:
#     st.info('Загрузите аудио')
# else:
#     st.audio(mnist_audio)
#
#     if st.button('Распознать'):
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
#                 tmp_file.write(mnist_audio.read())
#                 tmp_path = tmp_file.name
#
#             waveform, sample_rate = librosa.load(tmp_path, sr=22050)
#             waveform = torch.from_numpy(waveform).unsqueeze(0)
#             os.unlink(tmp_path)
#
#             mel_spec = transform(waveform)
#             mel_spec = mel_spec.mean(dim=0) if mel_spec.dim() == 3 else mel_spec
#             mel_spec = mel_spec.unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 y_prediction = model(mel_spec)
#                 prediction = y_prediction.argmax(dim=1).item()
#
#             st.success(f'Модель думает, что это: {classes[prediction]}')
#
#         except Exception as e:
#             st.error(f'Ошибка: {str(e)}')
