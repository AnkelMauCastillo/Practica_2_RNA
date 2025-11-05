# src/mlp.py

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, inicializacion='xavier'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialización de pesos
        if inicializacion == 'xavier':
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        elif inicializacion == 'normal':
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01

        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, lr):
        m = X.shape[0]

        # Gradientes de la capa de salida
        dZ2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Gradientes de la capa oculta
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.a1 * (1 - self.a1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Actualización de pesos
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)