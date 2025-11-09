import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import KFold
import time

def evaluar_modelo_gpu(model, X, y):
    """Evaluación completa del modelo en GPU"""
    model.eval()
    
    with torch.no_grad():
        probas = model.predict_proba(X)
        preds = (probas > 0.5).astype(int).flatten()
        y_flat = y.flatten()
    
    precision = precision_score(y_flat, preds, zero_division=0)
    recall = recall_score(y_flat, preds, zero_division=0)
    f1 = f1_score(y_flat, preds, zero_division=0)
    accuracy = accuracy_score(y_flat, preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'predictions': preds,
        'probabilities': probas
    }

def validacion_cruzada_gpu(config, X, y, k_folds=5, epochs=100, lr=0.01, batch_size=32):
    """Validación cruzada para las mejores configuraciones"""
    print(f"   🔍 Realizando validación cruzada ({k_folds}-folds)...")
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_scores = []
    fold_times = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        fold_start = time.time()
        
        # Dividir datos
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Crear y entrenar modelo
        modelo = MLP_GPU(
            input_size=X_train.shape[1],
            hidden_size=config['neuronas_ocultas'],
            inicializacion=config['inicializacion']
        )
        
        # Entrenar
        _ = entrenar_mlp_gpu(
            modelo, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, lr=lr
        )
        
        # Evaluar
        resultados = evaluar_modelo_gpu(modelo, X_val, y_val)
        fold_scores.append(resultados)
        fold_times.append(time.time() - fold_start)
        
        print(f"      Fold {fold+1}: F1 = {resultados['f1']:.4f}, "
              f"Precision = {resultados['precision']:.4f}, "
              f"Recall = {resultados['recall']:.4f}")
        
        # Liberar memoria
        del modelo
        torch.cuda.empty_cache()
    
    # Calcular promedios
    avg_scores = {
        'precision': np.mean([s['precision'] for s in fold_scores]),
        'recall': np.mean([s['recall'] for s in fold_scores]),
        'f1': np.mean([s['f1'] for s in fold_scores]),
        'accuracy': np.mean([s['accuracy'] for s in fold_scores]),
        'std_f1': np.std([s['f1'] for s in fold_scores]),
        'tiempo_promedio': np.mean(fold_times)
    }
    
    return avg_scores, fold_scores