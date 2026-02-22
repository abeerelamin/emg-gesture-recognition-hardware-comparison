import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm(input_shape, num_classes):
    model = models.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_gru(input_shape, num_classes):
    model = models.Sequential([
        layers.GRU(64, input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
