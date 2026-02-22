import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def create_windows(data, window_size=200, overlap=0.5):
    step = int(window_size * (1 - overlap))
    windows = []
    labels = []

    for i in range(0, len(data) - window_size, step):
        window = data[i:i + window_size]
        if len(set(window[:, -1])) == 1:
            windows.append(window[:, :-1])
            labels.append(window[0, -1])

    return np.array(windows), np.array(labels)


def split_and_scale(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)

    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    return X_train, X_test, y_train, y_test
