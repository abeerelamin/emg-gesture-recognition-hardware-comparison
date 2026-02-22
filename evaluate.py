from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    y_pred = model.predict(X_test).argmax(axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
