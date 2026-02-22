import time


def train_model(model, X_train, y_train, epochs=20, batch_size=32):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    start = time.time()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )

    end = time.time()

    print(f"Training time: {end - start:.2f} seconds")

    return history
