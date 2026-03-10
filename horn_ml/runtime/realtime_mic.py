import numpy as np
import librosa
import sounddevice as sd
import joblib
from collections import deque
import time

# ================= CONFIG =================
SR = 16000
WIN_SEC = 0.2
STEP_SEC = 0.1

WIN = int(WIN_SEC * SR)
STEP = int(STEP_SEC * SR)

MODEL_PATH = "../training/models/horn_rf.pkl"

PRINT_RMS = True          # показывать уровень сигнала
DETECT_STREAK = 3         # сколько подряд предсказаний = гудок
SILENCE_TIMEOUT = 0.1     # через сколько секунд считаем что гудок закончился
# ==========================================


# ---------- признаки ----------
def extract_features(chunk):
    mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=chunk, sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=SR)
    flatness = librosa.feature.spectral_flatness(y=chunk)
    return np.hstack([mfcc.mean(axis=1), centroid.mean(), bandwidth.mean(), flatness.mean()])


# ---------- загрузка модели ----------
print("Загрузка модели...")
model = joblib.load(MODEL_PATH)
print("Модель загружена")

# ---------- буфер ----------
audio_buffer = deque(maxlen=SR * 5)

detect_counter = 0
last_detect_time = 0
horn_active = False


# ---------- callback ----------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    audio_buffer.extend(indata[:, 0])


# ---------- старт микрофона ----------
print("\nОткрытие микрофона...")
stream = sd.InputStream(
    samplerate=SR,
    channels=1,
    dtype='float32',
    blocksize=STEP,
    callback=audio_callback
)
stream.start()

print("Микрофон запущен\n")

# ---------- основной цикл ----------
try:
    while True:

        if len(audio_buffer) < WIN:
            continue

        chunk = np.array(list(audio_buffer)[-WIN:])

        # ======= уровень сигнала =======
        rms = np.sqrt(np.mean(chunk**2))

        if PRINT_RMS:
            print(f"RMS: {rms:.4f}", end="\r")

        # слишком тихо → сразу фон
        if rms < 0.005:
            detect_counter = 0
            if horn_active and time.time() - last_detect_time > SILENCE_TIMEOUT:
                print("\n--- horn ended ---")
                horn_active = False
            continue

        # ======= предсказание =======
        X = extract_features(chunk).reshape(1, -1)
        pred = model.predict(X)[0]

        # ======= логика устойчивости =======
        if pred == 1:
            detect_counter += 1
            last_detect_time = time.time()
        else:
            detect_counter = max(0, detect_counter - 1)

        # ======= старт гудка =======
        if detect_counter >= DETECT_STREAK and not horn_active:
            print("\n>>> HORN DETECTED <<<")
            horn_active = True

        # ======= конец гудка =======
        if horn_active and time.time() - last_detect_time > SILENCE_TIMEOUT:
            print("\n--- horn ended ---")
            horn_active = False

except KeyboardInterrupt:
    print("\nОстановка")

finally:
    stream.stop()
    stream.close()
