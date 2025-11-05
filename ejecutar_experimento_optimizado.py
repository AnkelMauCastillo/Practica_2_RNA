# ejecutar_experimento_optimizado.py

import json
import numpy as np
import os
import time
import multiprocessing as mp
from src.preprocesamiento import Preprocesador
from src.representaciones import crear_vectorizador
from src.mlp import MLP
from src.entrenamiento import entrenar_mlp, evaluar_modelo
from configs import CONFIGURACIONES

def preprocesar_lote(config, textos, idioma='es'):
    """Preprocesa un lote de textos (reutilizable)"""
    preprocesador = Preprocesador(idioma=idioma)
    
    preprocesamiento_type = config.get('preprocesamiento', 'normalizar')
    
    if preprocesamiento_type == 'normalizar':
        usar_stopwords, usar_stemming = False, False
    elif preprocesamiento_type == 'normalizar_sin_stopwords':
        usar_stopwords, usar_stemming = True, False
    elif preprocesamiento_type == 'normalizar_sin_stopwords_stemming':
        usar_stopwords, usar_stemming = True, True
    else:
        usar_stopwords, usar_stemming = False, False
    
    return [preprocesador.preprocesar(t, usar_stopwords, usar_stemming) for t in textos]

def ejecutar_configuracion(config_idx, config, X_ent, y_ent, X_pru, y_pru):
    """Ejecuta una configuración individual"""
    try:
        inicio = time.time()
        
        # Preprocesamiento
        X_ent_limpio = preprocesar_lote(config, X_ent)
        X_pru_limpio = preprocesar_lote(config, X_pru)
        
        # Vectorización
        vectorizador = crear_vectorizador(config['pesado_terminos'], config['ngramas'])
        X_ent_vec = vectorizador.fit_transform(X_ent_limpio).toarray()
        X_pru_vec = vectorizador.transform(X_pru_limpio).toarray()
        
        # Modelo
        modelo = MLP(
            input_size=X_ent_vec.shape[1],
            hidden_size=config['neuronas_ocultas'],
            output_size=1,
            inicializacion=config['inicializacion']
        )
        
        y_ent_arr = np.array(y_ent).reshape(-1, 1)
        y_pru_arr = np.array(y_pru).reshape(-1, 1)
        
        # Entrenamiento
        train_losses, test_losses = entrenar_mlp(
            modelo, X_ent_vec, y_ent_arr, X_pru_vec, y_pru_arr,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )
        
        # Evaluación
        precision, recall, f1 = evaluar_modelo(modelo, X_pru_vec, y_pru_arr)
        tiempo = time.time() - inicio
        
        return {
            'config_idx': config_idx,
            'config': config,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'tiempo': tiempo,
            'exitoso': True
        }
        
    except Exception as e:
        return {
            'config_idx': config_idx,
            'config': config,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'train_losses': [],
            'test_losses': [],
            'tiempo': 0,
            'exitoso': False,
            'error': str(e)
        }

def main():
    print("🚀 EXPERIMENTO OPTIMIZADO - PRÁCTICA 2")
    
    # Cargar datos una sola vez
    print("📂 Cargando datos...")
    def cargar_datos(archivo):
        with open(archivo, 'r', encoding='utf-8') as f:
            contenido = f.read().strip()
            if contenido.startswith('['):
                datos = json.loads(contenido)
            else:
                datos = [json.loads(linea) for linea in f if linea.strip()]
        return [d.get('text', '') for d in datos], [d.get('klass', 0) for d in datos]
    
    X_ent, y_ent = cargar_datos('data/hateval_es_train.json')
    X_pru, y_pru = cargar_datos('data/hateval_es_test.json')
    
    print(f"✅ Datos cargados: {len(X_ent)} train, {len(X_pru)} test")
    
    # Ejecutar configuraciones
    resultados = []
    total_configs = len(CONFIGURACIONES)
    
    print(f"🔬 Ejecutando {total_configs} configuraciones...")
    print("⏳ Esto puede tomar varios minutos...")
    
    inicio_total = time.time()
    
    for i, config in enumerate(CONFIGURACIONES):
        print(f"\n[{i+1}/{total_configs}] Probando: {config}")
        
        resultado = ejecutar_configuracion(i, config, X_ent, y_ent, X_pru, y_pru)
        resultados.append(resultado)
        
        if resultado['exitoso']:
            print(f"   ✅ F1: {resultado['f1']:.4f}, Tiempo: {resultado['tiempo']:.1f}s")
        else:
            print(f"   ❌ Error: {resultado.get('error', 'Desconocido')}")
    
    tiempo_total = time.time() - inicio_total
    
    # Guardar resultados
    os.makedirs('resultados', exist_ok=True)
    with open('resultados/resultados_optimizados.txt', 'w', encoding='utf-8') as f:
        f.write("RESULTADOS OPTIMIZADOS - PRÁCTICA 2\n")
        f.write("="*80 + "\n\n")
        
        exitosos = [r for r in resultados if r['exitoso']]
        f.write(f"Configuraciones exitosas: {len(exitosos)}/{total_configs}\n")
        f.write(f"Tiempo total: {tiempo_total:.1f} segundos\n\n")
        
        for res in resultados:
            f.write(f"Config {res['config_idx']+1}: {res['config']}\n")
            if res['exitoso']:
                f.write(f"  F1: {res['f1']:.4f}, Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}\n")
                f.write(f"  Tiempo: {res['tiempo']:.1f}s\n")
            else:
                f.write(f"  ERROR: {res.get('error', 'Desconocido')}\n")
            f.write("-" * 60 + "\n")
    
    # Mostrar resumen
    print(f"\n🎉 EXPERIMENTO COMPLETADO")
    print(f"⏱️  Tiempo total: {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} minutos)")
    
    exitosos = [r for r in resultados if r['exitoso']]
    if exitosos:
        mejor = max(exitosos, key=lambda x: x['f1'])
        print(f"🏆 Mejor configuración: F1 = {mejor['f1']:.4f}")
        print(f"   Configuración: {mejor['config']}")

if __name__ == '__main__':
    main()