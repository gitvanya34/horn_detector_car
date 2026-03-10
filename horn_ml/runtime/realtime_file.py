import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- Настройки ----------------
SR = 16000
WIN_SEC = 0.2
STEP_SEC = 0.1
WIN = int(WIN_SEC * SR)
STEP = int(STEP_SEC * SR)

AUDIO_FILE = "../../data/raw/horn/freesound_community-044197_2013-dodge-charger-car-horn-wav-72822.mp3"
MODEL_FILE = "../training/models/horn_rf.pkl"

# ---------------- Функция признаков ----------------
def extract_features(chunk):
    mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=chunk, sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=SR)
    flatness = librosa.feature.spectral_flatness(y=chunk)
    return np.hstack([mfcc.mean(axis=1), centroid.mean(), bandwidth.mean(), flatness.mean()])

# ---------------- Загрузка аудио и модели ----------------
audio, _ = librosa.load(AUDIO_FILE, sr=SR)
model = joblib.load(MODEL_FILE)

# ---------------- Нарезка на окна ----------------
windows = []
for i in range(0, len(audio) - WIN, STEP):
    windows.append(audio[i:i+WIN])

# ---------------- Предсказания ----------------
predictions = []
for chunk in windows:
    X = extract_features(chunk).reshape(1, -1)
    pred = model.predict(X)[0]
    predictions.append(pred)


# ---------------- Подготовка сигнала гудка ----------------
time_axis = np.arange(len(audio)) / SR
pred_signal = np.zeros(len(audio))
for i, p in enumerate(predictions):
    if p:
        pred_signal[i*STEP:i*STEP+WIN] = 1  # 1 если гудок

# ---------------- Визуализация: два графика с разной высотой и цифрами ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,6), sharex=True,
                               gridspec_kw={'height_ratios':[2,1]})  # верх в 2 раза больше

# --- Спектрограмма ---
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
librosa.display.specshow(D, sr=SR, x_axis='time', y_axis='log', ax=ax1)
ax1.set_title("Спектрограмма (без colorbar)")

# --- График гудка снизу ---
ax2.fill_between(time_axis, 0, pred_signal, color='red', alpha=0.6)
ax2.set_ylim(0,1.1)
ax2.set_title("Гудки (1 = обнаружен)")
ax2.set_ylabel("HORN")
ax2.set_xlabel("Время (с)")

# --- Цифры на оси времени ---
max_time = len(audio)/SR
step = 1  # шаг для отметок, 1 секунда
ticks = np.arange(0, max_time+step, step)
ax2.set_xticks(ticks)
ax2.set_xticklabels([f"{t:.0f}" for t in ticks])

plt.tight_layout()
plt.show()
