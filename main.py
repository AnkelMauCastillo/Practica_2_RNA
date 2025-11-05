# main_completo.py

import json
import numpy as np
import os
import time
from src.preprocesamiento import Preprocesador
from src.representaciones import crear_vectorizador
from src.mlp import MLP
from src.entrenamiento import entrenar_mlp, evaluar_modelo
from src.visualizacion import graficar_perdidas, graficar_metricas_comparativas, generar_tabla_resultados
from configs import CONFIGURACIONES

def cargar_datos(archivo):
    """Carga datos desde archivo JSON o JSONL"""
    datos = []
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read().strip()
        
        if contenido.startswith('['):
            datos = json.loads(contenido)
        else:
            f.seek(0)
            for linea in f:
                linea = linea.strip()
                if linea:
                    try:
                        datos.append(json.loads(linea))
                    except json.JSONDecodeError:
                        continue
    
    textos = [d.get('text', '') for d in datos]
    etiquetas = [d.get('klass', 0) for d in datos]
    return textos, etiquetas

def aplicar_preprocesamiento(config, textos, idioma='es'):
    """Aplica el preprocesamiento según la configuración"""
    preprocesador = Preprocesador(idioma=idioma)
    
    preprocesamiento_type = config.get('preprocesamiento', 'normalizar')
    
    if preprocesamiento_type == 'normalizar':
        usar_stopwords = False
        usar_stemming = False
    elif preprocesamiento_type == 'normalizar_sin_stopwords':
        usar_stopwords = True
        usar_stemming = False
    elif preprocesamiento_type == 'normalizar_sin_stopwords_stemming':
        usar_stopwords = True
        usar_stemming = True
    else:
        usar_stopwords = False
        usar_stemming = False
    
    textos_procesados = []
    for i, texto in enumerate(textos):
        if i % 1000 == 0 and i > 0:
            print(f"  Preprocesados {i}/{len(textos)} textos...")
        texto_procesado = preprocesador.preprocesar(
            texto, 
            usar_stopwords=usar_stopwords, 
            usar_stemming=usar_stemming
        )
        textos_procesados.append(texto_procesado)
    
    return textos_procesados

def main():
    print("🚀 INICIANDO PRÁCTICA 2: CLASIFICACIÓN DE HATE SPEECH")
    print("📊 Probando todas las configuraciones de la Tabla 1")
    
    # Crear directorios
    os.makedirs('resultados/graficas', exist_ok=True)
    
    # Limpiar archivo de resultados
    with open('resultados/metricas_completas.txt', 'w', encoding='utf-8') as f:
        f.write("RESULTADOS COMPLETOS - PRÁCTICA 2\n")
        f.write("="*80 + "\n\n")
    
    try:
        # Cargar datos
        print("📂 Cargando datos...")
        X_entrenamiento, y_entrenamiento = cargar_datos('data/hateval_es_train.json')
        X_prueba, y_prueba = cargar_datos('data/hateval_es_test.json')
        
        print(f"✅ Datos cargados:")
        print(f"   - Entrenamiento: {len(X_entrenamiento)} ejemplos")
        print(f"   - Prueba: {len(X_prueba)} ejemplos")
        print(f"   - Distribución clases (train): {np.bincount(y_entrenamiento)}")
        print(f"   - Distribución clases (test): {np.bincount(y_prueba)}")
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return

    # Probar todas las configuraciones
    resultados = []
    tiempos_ejecucion = []
    
    print(f"\n🔬 Probando {len(CONFIGURACIONES)} configuraciones...")
    
    for config_idx, config in enumerate(CONFIGURACIONES):
        inicio_tiempo = time.time()
        
        print(f"\n{'='*80}")
        print(f"⚙️  Configuración {config_idx + 1}/{len(CONFIGURACIONES)}")
        print(f"{config}")
        print(f"{'='*80}")

        try:
            # Preprocesamiento
            print("🔄 Preprocesando textos...")
            X_ent_limpio = aplicar_preprocesamiento(config, X_entrenamiento, 'es')
            X_prueba_limpio = aplicar_preprocesamiento(config, X_prueba, 'es')

            # Vectorización
            print("📊 Vectorizando textos...")
            vectorizador = crear_vectorizador(
                tipo=config['pesado_terminos'],
                ngram_range=config['ngramas']
            )
            X_ent_vec = vectorizador.fit_transform(X_ent_limpio).toarray()
            X_prueba_vec = vectorizador.transform(X_prueba_limpio).toarray()

            print(f"   Dimensionalidad: {X_ent_vec.shape[1]} features")

            # Entrenar modelo
            modelo = MLP(
                input_size=X_ent_vec.shape[1],
                hidden_size=config['neuronas_ocultas'],
                output_size=1,
                inicializacion=config['inicializacion']
            )

            y_ent = np.array(y_entrenamiento).reshape(-1, 1)
            y_pru = np.array(y_prueba).reshape(-1, 1)

            print("🎯 Entrenando modelo...")
            train_losses, test_losses = entrenar_mlp(
                modelo, X_ent_vec, y_ent, X_prueba_vec, y_pru,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                lr=config['lr']
            )

            # Evaluar
            precision, recall, f1 = evaluar_modelo(modelo, X_prueba_vec, y_pru)
            
            tiempo_ejecucion = time.time() - inicio_tiempo
            tiempos_ejecucion.append(tiempo_ejecucion)
            
            resultado = {
                'config': config,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'tiempo_ejecucion': tiempo_ejecucion
            }
            resultados.append(resultado)
            
            print(f"\n📊 RESULTADOS:")
            print(f"   ✅ Precision: {precision:.4f}")
            print(f"   ✅ Recall: {recall:.4f}")
            print(f"   ✅ F1-score: {f1:.4f}")
            print(f"   ⏱️  Tiempo ejecución: {tiempo_ejecucion:.2f} segundos")

            # Guardar resultados detallados
            with open('resultados/metricas_completas.txt', 'a', encoding='utf-8') as f:
                f.write(f"CONFIGURACIÓN {config_idx + 1}:\n")
                f.write(f"  Parámetros: {config}\n")
                f.write(f"  Resultados: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n")
                f.write(f"  Tiempo: {tiempo_ejecucion:.2f} segundos\n")
                f.write("-" * 60 + "\n")

        except Exception as e:
            print(f"❌ Error en configuración {config_idx + 1}: {e}")
            # Agregar resultado vacío para mantener índices
            resultados.append({
                'config': config,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'train_losses': [],
                'test_losses': [],
                'tiempo_ejecucion': 0
            })

    # Análisis final
    print(f"\n{'='*80}")
    print("🎉 EXPERIMENTO COMPLETADO")
    print(f"{'='*80}")
    
    # Generar gráficas y tablas
    print("\n📈 Generando gráficas y tablas...")
    graficar_perdidas(CONFIGURACIONES, resultados, top_n=5)
    graficar_metricas_comparativas(resultados, CONFIGURACIONES)
    generar_tabla_resultados(resultados, CONFIGURACIONES)
    
    # Mejores configuraciones
    mejores_indices = np.argsort([r['f1'] for r in resultados])[-3:][::-1]
    
    print(f"\n🏆 MEJORES 3 CONFIGURACIONES (por F1-score):")
    for i, idx in enumerate(mejores_indices):
        res = resultados[idx]
        print(f"  {i+1}. Config {idx+1}: F1 = {res['f1']:.4f}, Precision = {res['precision']:.4f}, Recall = {res['recall']:.4f}")
    
    print(f"\n⏱️  Tiempo total de ejecución: {sum(tiempos_ejecucion):.2f} segundos")
    print(f"📁 Resultados guardados en: /resultados/")

if __name__ == '__main__':
    main()