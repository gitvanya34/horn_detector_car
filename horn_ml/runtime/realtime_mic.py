import numpy as np
import librosa
import sounddevice as sd
import joblib
from collections import deque
import time

# ================= CONFIG =================
SR = 16000         # частота модели
WIN_SEC = 0.2
STEP_SEC = 0.1

LED_PIN = 17       # GPIO для LED
PRINT_RMS = True
DETECT_STREAK = 3
SILENCE_TIMEOUT = 0.1

WIN = int(WIN_SEC * SR)
STEP = int(STEP_SEC * SR)

MODEL_PATH = "../training/models/horn_rf.pkl"

# ==========================================

# ---------- проверка GPIO ----------
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
except ImportError:
    GPIO_AVAILABLE = False

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
audio_buffer = deque(maxlen=int(SR*5))
detect_counter = 0
last_detect_time = 0
horn_active = False

# ---------- выбираем USB микрофон ----------
def find_usb_mic():
    for i, dev in enumerate(sd.query_devices()):
        if 'usb' in dev['name'].lower() and dev['max_input_channels'] > 0:
            return i, dev['default_samplerate']
    # fallback на default
    dev = sd.query_devices(kind='input')
    return dev['index'], dev['default_samplerate']

MIC_DEVICE, MIC_SR = find_usb_mic()
MIC_SR = int(MIC_SR)
print(f"Using microphone: {MIC_DEVICE}, SR: {MIC_SR}")

# ---------- callback ----------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_buffer.extend(indata[:, 0])  # только первый канал

# ---------- старт микрофона ----------
print("\nОткрытие микрофона...")
stream = sd.InputStream(
    device=MIC_DEVICE,
    samplerate=MIC_SR,
    channels=1,
    dtype='float32',
    blocksize=int(MIC_SR * STEP_SEC),
    callback=audio_callback
)
stream.start()
print("Микрофон запущен\n")

# ---------- основной цикл ----------
try:
    while True:
        # ждем, пока накопится окно
        if len(audio_buffer) < int(WIN_SEC * MIC_SR):
            continue

        # берем последние WIN_SEC секунд из реального микрофона
        chunk_raw = np.array(list(audio_buffer)[-int(WIN_SEC * MIC_SR):])

        # ресемплируем в 16kHz для модели
        chunk = librosa.resample(chunk_raw, orig_sr=MIC_SR, target_sr=SR)

        # RMS
        rms = np.sqrt(np.mean(chunk**2))
        if PRINT_RMS:
            print(f"RMS: {rms:.4f}", end="\r")

        if rms < 0.005:
            detect_counter = 0
            if horn_active and time.time() - last_detect_time > SILENCE_TIMEOUT:
                print("\n--- horn ended ---")
                if GPIO_AVAILABLE:
                    GPIO.output(LED_PIN, GPIO.LOW)
                horn_active = False
            continue

        # предсказание
        X = extract_features(chunk).reshape(1, -1)
        pred = model.predict(X)[0]

        if pred == 1:
            detect_counter += 1
            last_detect_time = time.time()
        else:
            detect_counter = max(0, detect_counter - 1)

        # старт гудка
        if detect_counter >= DETECT_STREAK and not horn_active:
            print("\n>>> HORN DETECTED <<<")
            if GPIO_AVAILABLE:
                GPIO.output(LED_PIN, GPIO.HIGH)
            horn_active = True

        # конец гудка
        if horn_active and time.time() - last_detect_time > SILENCE_TIMEOUT:
            print("\n--- horn ended ---")
            if GPIO_AVAILABLE:
                GPIO.output(LED_PIN, GPIO.LOW)
            horn_active = False

except KeyboardInterrupt:
    print("\nОстановка")

finally:
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.LOW)
    stream.stop()
    stream.close()
    if GPIO_AVAILABLE:
        GPIO.cleanup()