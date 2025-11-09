import torch
import torch.nn as nn
import numpy as np
import time

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def entrenar_mlp_gpu(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=0.01, patience=15):
    """Entrenamiento optimizado para GPU con métricas completas"""
    
    device = model.device
    criterion = nn.BCELoss()  # Binary Cross Entropy para clasificación binaria
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convertir datos a tensores y mover a GPU
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Crear DataLoader para mini-batches
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    early_stopping = EarlyStopping(patience=patience)
    
    print(f"🚀 Iniciando entrenamiento en {device}...")
    print(f"   Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}")
    print(f"   Parámetros totales: {model.get_parameters_count():,}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Fase de entrenamiento
        model.train()
        epoch_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Calcular accuracy
            preds = (outputs > 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        # Fase de evaluación
        model.eval()
        with torch.no_grad():
            # Pérdida y accuracy en entrenamiento
            train_outputs = model(X_train_tensor)
            train_loss = criterion(train_outputs, y_train_tensor).item()
            train_preds = (train_outputs > 0.5).float()
            train_accuracy = (train_preds == y_train_tensor).float().mean().item()
            
            # Pérdida y accuracy en prueba
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            test_preds = (test_outputs > 0.5).float()
            test_accuracy = (test_preds == y_test_tensor).float().mean().item()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - epoch_start
        
        # Early stopping
        early_stopping(test_loss)
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"   Época {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f} | "
                  f"Tiempo: {epoch_time:.2f}s")
        
        if early_stopping.early_stop:
            print(f"   ⏹️  Early stopping en época {epoch}")
            break
    
    # Liberar memoria de GPU
    torch.cuda.empty_cache()
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }