import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from augment import augment_signal

# ---------------- Параметры ----------------
SR = 16000
WIN_SEC = 0.2
STEP_SEC = 0.1
WIN = int(WIN_SEC * SR)
STEP = int(STEP_SEC * SR)


# ---------- нарезка окон ----------
def make_windows(fragment):
    windows = []

    if len(fragment) < WIN:
        reps = int(np.ceil(WIN / len(fragment)))
        window = np.tile(fragment, reps)[:WIN]
        windows.append(window)
    else:
        for start in range(0, len(fragment) - WIN + 1, STEP):
            windows.append(fragment[start:start + WIN])

        tail = fragment[len(fragment) - WIN:]
        windows.append(tail)

    return windows


# ---------- признаки ----------
def extract_features(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SR)
    flatness = librosa.feature.spectral_flatness(y=y)

    return np.hstack([
        mfcc.mean(axis=1),
        centroid.mean(),
        bandwidth.mean(),
        flatness.mean()
    ])


# ---------- загрузка ----------
def load_fragments(folder):
    fragments = []

    for f in os.listdir(folder):
        if not f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            continue

        path = os.path.join(folder, f)
        print("Загрузка:", path)

        y, _ = librosa.load(path, sr=SR)
        fragments.append(y)

    return fragments


# ---------- формирование датасета ----------
def build_dataset(pos_folder='../../data/raw/clean_horn/pos',
                  neg_folder='../../data/raw/clean_horn/neg'):

    X = []
    y = []

    # ---------- POS (гудки) ----------
    pos_fragments = load_fragments(pos_folder)

    for i, frag in enumerate(pos_fragments):
        print(f"\nАугментация POS {i+1}/{len(pos_fragments)}")

        augmented_versions = augment_signal(frag, SR)

        for aug in augmented_versions:
            windows = make_windows(aug)

            for w in windows:
                X.append(extract_features(w))
                y.append(1)

    # ---------- NEG (фон) ----------
    neg_fragments = load_fragments(neg_folder)

    for frag in neg_fragments:
        windows = make_windows(frag)

        for w in windows:
            X.append(extract_features(w))
            y.append(0)

    X = np.array(X)
    y = np.array(y)

    print("\nИТОГО:")
    print("Гудки:", np.sum(y == 1))
    print("Фон :", np.sum(y == 0))

    return X, y


# ---------- обучение ----------
def train_model():

    X, y = build_dataset()

    print("\nРазмер датасета:", X.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        max_depth=12,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

    print("\nОбучение...")
    model.fit(X_train, y_train)

    print("\nВалидация:")

    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))


    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/horn_rf.pkl')

    print("\nМодель сохранена: models/horn_rf.pkl")



# ---------- запуск ----------
if __name__ == "__main__":
    train_model()
