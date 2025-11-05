# src/entrenamiento.py

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def entrenar_mlp(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=0.01):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Mini-batch
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward y backward
            output = model.forward(X_batch)
            model.backward(X_batch, y_batch, output, lr)

        # Pérdidas
        train_pred = model.forward(X_train)
        train_loss = model.compute_loss(y_train, train_pred)
        train_losses.append(train_loss)

        test_pred = model.forward(X_test)
        test_loss = model.compute_loss(y_test, test_pred)
        test_losses.append(test_loss)

        if epoch % 50 == 0:
            print(f"Época {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

    return train_losses, test_losses

def evaluar_modelo(model, X, y):
    preds = model.forward(X)
    preds_bin = (preds > 0.5).astype(int).flatten()
    precision = precision_score(y, preds_bin)
    recall = recall_score(y, preds_bin)
    f1 = f1_score(y, preds_bin)
    return precision, recall, f1