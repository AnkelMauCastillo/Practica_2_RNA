import json
import numpy as np
import os
import time
import torch
from src.preprocesamiento import Preprocesador
from src.representaciones import crear_vectorizador
from src.mlp_gpu import MLP_GPU
from src.entrenamiento_gpu import entrenar_mlp_gpu
from src.evaluacion_gpu import evaluar_modelo_gpu
from sklearn.model_selection import KFold

def validacion_cruzada_simple(config, X, y, k_folds=5, epochs=50):
    """Validación cruzada simplificada sin dependencias circulares"""
    print(f"   🔍 Validación cruzada ({k_folds}-folds, {epochs} épocas)...")
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        # Dividir datos
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Crear modelo
        modelo = MLP_GPU(
            input_size=X_train.shape[1],
            hidden_size=config['neuronas_ocultas'],
            inicializacion=config['inicializacion']
        )
        
        # Entrenar
        _ = entrenar_mlp_gpu(
            modelo, X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=min(config['batch_size'], 64),
            lr=config['lr'],
            patience=10
        )
        
        # Evaluar
        resultados = evaluar_modelo_gpu(modelo, X_val, y_val)
        fold_scores.append(resultados)
        
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
        'std_f1': np.std([s['f1'] for s in fold_scores])
    }
    
    return avg_scores, fold_scores

def ejecutar_validacion_cruzada_final():
    """Ejecuta solo la validación cruzada para las mejores configuraciones"""
    
    print("🎯 VALIDACIÓN CRUZADA FINAL - 5 FOLDS")
    print("="*60)
    
    # Configuraciones basadas en tus mejores resultados
    configs_mejores = {
        'es': [
            {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
             'ngramas': (1,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128},
            {'neuronas_ocultas': 64, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
             'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128},
            {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
             'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128}
        ],
        'en': [
            {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
             'ngramas': (1,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128},
            {'neuronas_ocultas': 64, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
             'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128},
            {'neuronas_ocultas': 1024, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
             'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32}
        ]
    }
    
    def cargar_datos_simple(archivo):
        """Carga datos simplificada"""
        with open(archivo, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        textos = [d.get('text', '') for d in datos]
        etiquetas = [d.get('klass', 0) for d in datos]
        return textos, etiquetas
    
    for idioma in ['es', 'en']:
        print(f"\n📊 PROCESANDO: {idioma.upper()}")
        
        try:
            # Cargar datos completos
            X_todos, y_todos = cargar_datos_simple(f'data/hateval_{idioma}_all.json')
            print(f"   ✅ Datos: {len(X_todos)} ejemplos")
            print(f"   📈 Distribución: {np.bincount(y_todos)}")
            
            for i, config in enumerate(configs_mejores[idioma]):
                print(f"\n   🔧 Configuración {i+1}/3:")
                print(f"      {config}")
                
                # Preprocesar
                preprocesador = Preprocesador(idioma=idioma)
                X_procesados = [preprocesador.preprocesar(texto) for texto in X_todos]
                
                # Vectorizar
                vectorizador = crear_vectorizador(
                    tipo=config['pesado_terminos'],
                    ngram_range=config['ngramas']
                )
                X_vec = vectorizador.fit_transform(X_procesados).toarray()
                y_vec = np.array(y_todos).reshape(-1, 1)
                
                print(f"      ✅ Vectorizado: {X_vec.shape}")
                
                # Validación cruzada
                avg_scores, fold_scores = validacion_cruzada_simple(
                    config, X_vec, y_vec, k_folds=5, epochs=50
                )
                
                print(f"      📊 Resultados CV:")
                print(f"        F1: {avg_scores['f1']:.4f} ± {avg_scores['std_f1']:.4f}")
                print(f"        Precision: {avg_scores['precision']:.4f}")
                print(f"        Recall: {avg_scores['recall']:.4f}")
                print(f"        Accuracy: {avg_scores['accuracy']:.4f}")
                
                # Guardar resultados
                with open(f'resultados/validacion_cruzada_final_{idioma}.txt', 'a', encoding='utf-8') as f:
                    f.write(f"CONFIGURACIÓN {i+1}:\n")
                    f.write(f"  Parámetros: {config}\n")
                    f.write(f"  F1: {avg_scores['f1']:.4f} ± {avg_scores['std_f1']:.4f}\n")
                    f.write(f"  Precision: {avg_scores['precision']:.4f}\n")
                    f.write(f"  Recall: {avg_scores['recall']:.4f}\n")
                    f.write(f"  Accuracy: {avg_scores['accuracy']:.4f}\n")
                    f.write("-" * 50 + "\n\n")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == '__main__':
    ejecutar_validacion_cruzada_final()