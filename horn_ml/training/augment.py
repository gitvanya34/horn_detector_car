import numpy as np
import librosa


def add_noise(y, level=0.01):
    noise = np.random.randn(len(y))
    return y + level * noise


def random_gain(y):
    gain = np.random.uniform(0.2, 1.8)
    return y * gain


def pitch_shift(y, sr):
    steps = np.random.uniform(-3, 3)  # разные машины
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)


def time_stretch_safe(y):
    rate = np.random.uniform(0.85, 1.15)
    try:
        y2 = librosa.effects.time_stretch(y, rate=rate)
        # возвращаем исходную длину
        if len(y2) > len(y):
            y2 = y2[:len(y)]
        else:
            y2 = np.pad(y2, (0, len(y)-len(y2)))
        return y2
    except:
        return y


def bandpass_simulation(y, sr):
    # имитация разных микрофонов/телефонов
    low = np.random.uniform(200, 600)
    high = np.random.uniform(2500, 5000)

    y = librosa.effects.preemphasis(y)

    S = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr)

    mask = (freqs > low) & (freqs < high)
    S[~mask, :] *= 0.2

    return librosa.istft(S, length=len(y))


def augment_signal(y, sr):

    augmented = []

    # оригинал
    augmented.append(y)

    # громкость
    for _ in range(3):
        augmented.append(random_gain(y))

    # шум
    for lvl in [0.005, 0.01, 0.02, 0.03]:
        augmented.append(add_noise(y, lvl))

    # высота
    for _ in range(3):
        augmented.append(pitch_shift(y, sr))

    # скорость
    for _ in range(3):
        augmented.append(time_stretch_safe(y))

    # микрофоны
    for _ in range(3):
        augmented.append(bandpass_simulation(y, sr))

    return augmented