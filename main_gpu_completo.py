import json
import numpy as np
import os
import time
import torch
from src.preprocesamiento import Preprocesador
from src.representaciones import crear_vectorizador
from src.mlp_gpu import MLP_GPU
from src.entrenamiento_gpu import entrenar_mlp_gpu
from src.evaluacion_gpu import evaluar_modelo_gpu, validacion_cruzada_gpu
from src.visualizacion import (graficar_perdidas, graficar_metricas_comparativas, 
                              generar_tabla_resultados, graficar_evolucion_entrenamiento,
                              graficar_top5_configuraciones, graficar_comparacion_es_en)
from configs_completas_gpu import CONFIGURACIONES


def ejecutar_validacion_cruzada_completa(idioma='es', k_folds=5):
    """Ejecuta validación cruzada para las 3 mejores configuraciones"""
    print(f"\n🎯 INICIANDO VALIDACIÓN CRUZADA ({k_folds}-folds) - {idioma.upper()}")
    
    try:
        # Cargar todos los datos
        X_todos, y_todos = cargar_datos(f'data/hateval_{idioma}_all.json')
        print(f"   📊 Datos cargados: {len(X_todos)} ejemplos")
        print(f"   📈 Distribución de clases: {np.bincount(y_todos)}")
        
        # Identificar las 3 mejores configuraciones basadas en resultados previos
        if idioma == 'es':
            mejores_config_indices = [8, 1, 2]  # Basado en F1-score más alto
            mejores_configs = [
                CONFIGURACIONES[8],  # Config 9: (1,2) n-grams
                CONFIGURACIONES[0],  # Config 1: 64 neuronas
                CONFIGURACIONES[1]   # Config 2: 128 neuronas
            ]
        else:  # 'en'
            mejores_config_indices = [8, 1, 5]  # Basado en F1-score más alto
            mejores_configs = [
                CONFIGURACIONES[8],  # Config 9: (1,2) n-grams
                CONFIGURACIONES[0],  # Config 1: 64 neuronas  
                CONFIGURACIONES[4]   # Config 5: 1024 neuronas
            ]
        
        resultados_cv = {}
        
        for i, (config_idx, config) in enumerate(zip(mejores_config_indices, mejores_configs)):
            print(f"\n   🔍 Configuración {i+1}/3 (Original: Config {config_idx+1}):")
            print(f"      Neuronas: {config['neuronas_ocultas']}, Inicial: {config['inicializacion']}")
            print(f"      Pesado: {config['pesado_terminos']}, Ngramas: {config['ngramas']}")
            print(f"      Preproc: {config['preprocesamiento']}, LR: {config['lr']}")
            
            # Preprocesar datos para esta configuración
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
            
            print(f"      Preprocesando textos...")
            X_procesados = []
            for texto in X_todos:
                texto_proc = preprocesador.preprocesar(
                    texto, 
                    usar_stopwords=usar_stopwords, 
                    usar_stemming=usar_stemming
                )
                X_procesados.append(texto_proc)
            
            # Vectorizar
            vectorizador = crear_vectorizador(
                tipo=config['pesado_terminos'],
                ngram_range=config['ngramas']
            )
            X_vec = vectorizador.fit_transform(X_procesados).toarray()
            y_vec = np.array(y_todos).reshape(-1, 1)
            
            print(f"      ✅ Datos vectorizados: {X_vec.shape}")
            
            # Ejecutar validación cruzada
            cv_start = time.time()
            avg_scores, fold_scores = validacion_cruzada_gpu(
                config, X_vec, y_vec, entrenar_mlp_gpu, k_folds=k_folds,  # <-- AGREGAR entrenar_mlp_gpu
                epochs=min(config['epochs'], 100),
                lr=config['lr'],
                batch_size=config['batch_size']
            )
            cv_time = time.time() - cv_start
            
            resultados_cv[config_idx] = {
                'config': config,
                'cv_scores': avg_scores,
                'fold_scores': fold_scores,
                'tiempo_total': cv_time
            }
            
            print(f"      📊 Resultados CV:")
            print(f"        F1: {avg_scores['f1']:.4f} ± {avg_scores['std_f1']:.4f}")
            print(f"        Precision: {avg_scores['precision']:.4f}")
            print(f"        Recall: {avg_scores['recall']:.4f}")
            print(f"        Accuracy: {avg_scores['accuracy']:.4f}")
            print(f"        Tiempo: {cv_time:.2f}s")
            
            # Liberar memoria
            del vectorizador
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Guardar resultados de validación cruzada
        guardar_resultados_validacion_cruzada(resultados_cv, idioma)
        return resultados_cv
        
    except Exception as e:
        print(f"   ❌ Error en validación cruzada: {e}")
        import traceback
        traceback.print_exc()
        return None

def guardar_resultados_validacion_cruzada(resultados_cv, idioma):
    """Guarda los resultados de validación cruzada en archivo"""
    with open(f'resultados/validacion_cruzada_{idioma}.txt', 'w', encoding='utf-8') as f:
        f.write(f"RESULTADOS VALIDACIÓN CRUZADA (5-folds) - {idioma.upper()}\n")
        f.write("="*80 + "\n\n")
        
        for config_idx, resultado in resultados_cv.items():
            config = resultado['config']
            scores = resultado['cv_scores']
            
            f.write(f"CONFIGURACIÓN ORIGINAL {config_idx+1}:\n")
            f.write(f"  Parámetros: {config}\n")
            f.write(f"  Resultados Validación Cruzada:\n")
            f.write(f"    F1-score: {scores['f1']:.4f} ± {scores['std_f1']:.4f}\n")
            f.write(f"    Precision: {scores['precision']:.4f}\n")
            f.write(f"    Recall: {scores['recall']:.4f}\n")
            f.write(f"    Accuracy: {scores['accuracy']:.4f}\n")
            f.write(f"    Tiempo promedio por fold: {scores['tiempo_promedio']:.2f}s\n")
            f.write(f"    Tiempo total validación: {resultado['tiempo_total']:.2f}s\n")
            
            # Resultados por fold
            f.write(f"  Resultados por Fold:\n")
            for fold_idx, fold_score in enumerate(resultado['fold_scores']):
                f.write(f"    Fold {fold_idx+1}: F1={fold_score['f1']:.4f}, "
                       f"Precision={fold_score['precision']:.4f}, "
                       f"Recall={fold_score['recall']:.4f}, "
                       f"Accuracy={fold_score['accuracy']:.4f}\n")
            
            f.write("-" * 80 + "\n\n")

def generar_analisis_comparativo(resultados_es, resultados_en):
    """Genera análisis comparativo automático basado en resultados"""
    
    # Encontrar mejores configuraciones por idioma
    mejor_es_idx = np.argmax([r['f1'] for r in resultados_es])
    mejor_en_idx = np.argmax([r['f1'] for r in resultados_en])
    
    mejor_es = resultados_es[mejor_es_idx]
    mejor_en = resultados_en[mejor_en_idx]
    
    with open('resultados/analisis_comparativo.txt', 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS COMPARATIVO - RESULTADOS EXPERIMENTALES\n")
        f.write("="*80 + "\n\n")
        
        f.write("MEJORES CONFIGURACIONES POR IDIOMA:\n")
        f.write("-" * 50 + "\n")
        f.write(f"ESPAÑOL - Config {mejor_es_idx+1}:\n")
        f.write(f"  F1: {mejor_es['f1']:.4f}, Precision: {mejor_es['precision']:.4f}, Recall: {mejor_es['recall']:.4f}\n")
        f.write(f"  Parámetros: {mejor_es['config']}\n\n")
        
        f.write(f"INGLÉS - Config {mejor_en_idx+1}:\n")
        f.write(f"  F1: {mejor_en['f1']:.4f}, Precision: {mejor_en['precision']:.4f}, Recall: {mejor_en['recall']:.4f}\n")
        f.write(f"  Parámetros: {mejor_en['config']}\n\n")
        
        f.write("COMPARATIVO ENTRE IDIOMAS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Diferencia en F1-score: {mejor_es['f1'] - mejor_en['f1']:.4f}\n")
        f.write(f"Rendimiento relativo: {(mejor_es['f1']/mejor_en['f1']-1)*100:+.1f}%\n\n")
        
        f.write("TENDENCIAS OBSERVADAS:\n")
        f.write("-" * 50 + "\n")
        
        # Análisis de neuronas
        f.write("1. NEURONAS OCULTAS:\n")
        neuronas_es = [r for r in resultados_es if r['config']['ngramas'] == (1,1) and r['config']['preprocesamiento'] == 'normalizar']
        neuronas_en = [r for r in resultados_en if r['config']['ngramas'] == (1,1) and r['config']['preprocesamiento'] == 'normalizar']
        
        for i, (res_es, res_en) in enumerate(zip(neuronas_es[:5], neuronas_en[:5])):
            f.write(f"  {res_es['config']['neuronas_ocultas']} neuronas - ES: {res_es['f1']:.4f}, EN: {res_en['f1']:.4f}\n")
        
        # Análisis de n-gramas
        f.write("\n2. N-GRAMAS:\n")
        ngrams_es = [r for r in resultados_es if r['config']['neuronas_ocultas'] == 128 and r['config']['preprocesamiento'] == 'normalizar']
        ngrams_en = [r for r in resultados_en if r['config']['neuronas_ocultas'] == 128 and r['config']['preprocesamiento'] == 'normalizar']
        
        for res in ngrams_es:
            if res['config']['ngramas'] in [(1,1), (2,2), (1,2)]:
                f.write(f"  {res['config']['ngramas']} - ES: {res['f1']:.4f}\n")
        for res in ngrams_en:
            if res['config']['ngramas'] in [(1,1), (2,2), (1,2)]:
                f.write(f"  {res['config']['ngramas']} - EN: {res['f1']:.4f}\n")

def verificar_gpu():
    """Verifica y muestra información de la GPU"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPUs disponibles: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Configurar GPU por defecto
        torch.cuda.set_device(0)
        print(f"   Usando GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("❌ No se encontró GPU compatible con CUDA")
        return False

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
    
    print(f"   Preprocesamiento: {preprocesamiento_type}")
    textos_procesados = []
    for i, texto in enumerate(textos):
        if i % 1000 == 0 and i > 0:
            print(f"     Procesados {i}/{len(textos)} textos...")
        texto_procesado = preprocesador.preprocesar(
            texto, 
            usar_stopwords=usar_stopwords, 
            usar_stemming=usar_stemming
        )
        textos_procesados.append(texto_procesado)
    
    return textos_procesados

def ejecutar_experimento_idioma(idioma='es'):
    """Ejecuta el experimento completo para un idioma"""
    print(f"\n{'='*80}")
    print(f"🏁 INICIANDO EXPERIMENTO PARA IDIOMA: {idioma.upper()}")
    print(f"{'='*80}")
    
    # Cargar datos
    print("📂 Cargando datos...")
    try:
        X_entrenamiento, y_entrenamiento = cargar_datos(f'data/hateval_{idioma}_train.json')
        X_prueba, y_prueba = cargar_datos(f'data/hateval_{idioma}_test.json')
        
        print(f"   ✅ Datos cargados:")
        print(f"      - Entrenamiento: {len(X_entrenamiento)} ejemplos")
        print(f"      - Prueba: {len(X_prueba)} ejemplos")
        print(f"      - Distribución clases (train): {np.bincount(y_entrenamiento)}")
        print(f"      - Distribución clases (test): {np.bincount(y_prueba)}")
        
    except Exception as e:
        print(f"   ❌ Error cargando datos: {e}")
        return []

    # Probar todas las configuraciones
    resultados = []
    tiempos_ejecucion = []
    
    print(f"\n🔬 Probando {len(CONFIGURACIONES)} configuraciones para {idioma.upper()}...")
    
    for config_idx, config in enumerate(CONFIGURACIONES):
        inicio_tiempo = time.time()
        
        print(f"\n{'─'*80}")
        print(f"  Configuración {config_idx + 1}/{len(CONFIGURACIONES)}")
        print(f"  Neuronas: {config['neuronas_ocultas']}, Inicial: {config['inicializacion']}")
        print(f"  Pesado: {config['pesado_terminos']}, Ngramas: {config['ngramas']}")
        print(f"  Preproc: {config['preprocesamiento']}, LR: {config['lr']}, Batch: {config['batch_size']}")
        print(f"{'─'*80}")

        try:
            # Preprocesamiento
            print("  🧹 Preprocesando textos...")
            X_ent_limpio = aplicar_preprocesamiento(config, X_entrenamiento, idioma)
            X_prueba_limpio = aplicar_preprocesamiento(config, X_prueba, idioma)

            # Vectorización
            print("  🔢 Vectorizando textos...")
            vectorizador = crear_vectorizador(
                tipo=config['pesado_terminos'],
                ngram_range=config['ngramas']
            )
            X_ent_vec = vectorizador.fit_transform(X_ent_limpio).toarray()
            X_prueba_vec = vectorizador.transform(X_prueba_limpio).toarray()

            print(f"     ✅ Dimensionalidad: {X_ent_vec.shape[1]} features")
            print(f"     ✅ Memoria: {(X_ent_vec.nbytes + X_prueba_vec.nbytes) / 1024**2:.2f} MB")

            # Crear y entrenar modelo en GPU
            modelo = MLP_GPU(
                input_size=X_ent_vec.shape[1],
                hidden_size=config['neuronas_ocultas'],
                inicializacion=config['inicializacion']
            )

            y_ent = np.array(y_entrenamiento).reshape(-1, 1)
            y_pru = np.array(y_prueba).reshape(-1, 1)

            print("  🎯 Entrenando modelo en GPU...")
            metricas_entrenamiento = entrenar_mlp_gpu(
                modelo, X_ent_vec, y_ent, X_prueba_vec, y_pru,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                lr=config['lr']
            )

            # Evaluar
            resultados_evaluacion = evaluar_modelo_gpu(modelo, X_prueba_vec, y_pru)
            
            tiempo_ejecucion = time.time() - inicio_tiempo
            tiempos_ejecucion.append(tiempo_ejecucion)
            
            resultado = {
                'config': config,
                'precision': resultados_evaluacion['precision'],
                'recall': resultados_evaluacion['recall'],
                'f1': resultados_evaluacion['f1'],
                'accuracy': resultados_evaluacion['accuracy'],
                'train_losses': metricas_entrenamiento['train_losses'],
                'test_losses': metricas_entrenamiento['test_losses'],
                'train_accuracies': metricas_entrenamiento['train_accuracies'],
                'test_accuracies': metricas_entrenamiento['test_accuracies'],
                'tiempo_ejecucion': tiempo_ejecucion
            }
            resultados.append(resultado)
            
            print(f"\n  📊 RESULTADOS:")
            print(f"     ✅ Precision: {resultados_evaluacion['precision']:.4f}")
            print(f"     ✅ Recall: {resultados_evaluacion['recall']:.4f}")
            print(f"     ✅ F1-score: {resultados_evaluacion['f1']:.4f}")
            print(f"     ✅ Accuracy: {resultados_evaluacion['accuracy']:.4f}")
            print(f"     ⏱️  Tiempo: {tiempo_ejecucion:.2f} segundos")

            # Guardar resultados detallados
            with open(f'resultados/metricas_detalladas_{idioma}.txt', 'a', encoding='utf-8') as f:
                f.write(f"CONFIGURACIÓN {config_idx + 1}:\n")
                f.write(f"  Parámetros: {config}\n")
                f.write(f"  Resultados: Precision={resultados_evaluacion['precision']:.4f}, Recall={resultados_evaluacion['recall']:.4f}, F1={resultados_evaluacion['f1']:.4f}, Accuracy={resultados_evaluacion['accuracy']:.4f}\n")
                f.write(f"  Tiempo: {tiempo_ejecucion:.2f} segundos\n")
                f.write("-" * 60 + "\n")

            # Liberar memoria GPU después de cada configuración
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ Error en configuración {config_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            
            # Agregar resultado vacío para mantener índices
            resultados.append({
                'config': config,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'accuracy': 0,
                'train_losses': [],
                'test_losses': [],
                'train_accuracies': [],
                'test_accuracies': [],
                'tiempo_ejecucion': 0
            })

    return resultados, tiempos_ejecucion

def ejecutar_validacion_cruzada_mejores(idioma='es', resultados=None, k_folds=5):
    """Ejecuta validación cruzada para las 3 mejores configuraciones"""
    if resultados is None:
        return
    
    # Obtener las 3 mejores configuraciones por F1-score
    mejores_indices = np.argsort([r['f1'] for r in resultados])[-3:][::-1]
    
    print(f"\n🎯 EJECUTANDO VALIDACIÓN CRUZADA PARA LAS 3 MEJORES CONFIGURACIONES ({idioma.upper()})")
    
    # Cargar todos los datos para validación cruzada
    try:
        X_todos, y_todos = cargar_datos(f'data/hateval_{idioma}_all.json')
        print(f"   📊 Datos para validación cruzada: {len(X_todos)} ejemplos")
        
        # Preprocesar y vectorizar (usar la primera configuración como base)
        config_base = CONFIGURACIONES[0]
        preprocesador = Preprocesador(idioma=idioma)
        
        X_procesados = []
        for texto in X_todos:
            texto_proc = preprocesador.preprocesar(texto, usar_stopwords=False, usar_stemming=False)
            X_procesados.append(texto_proc)
        
        vectorizador = crear_vectorizador(tipo='tf', ngram_range=(1,1))
        X_vec = vectorizador.fit_transform(X_procesados).toarray()
        y_vec = np.array(y_todos).reshape(-1, 1)
        
        print(f"   ✅ Datos vectorizados: {X_vec.shape}")
        
        resultados_cv = []
        
        for i, idx in enumerate(mejores_indices):
            config = CONFIGURACIONES[idx]
            resultado_original = resultados[idx]
            
            print(f"\n   🔍 Configuración {idx+1} (Top {i+1}):")
            print(f"      Parámetros: {config}")
            print(f"      F1 original: {resultado_original['f1']:.4f}")
            
            # Ejecutar validación cruzada
            avg_scores, fold_scores = validacion_cruzada_gpu(
                config, X_vec, y_vec, k_folds=k_folds,
                epochs=min(config['epochs'], 100),  # Reducir epochs para CV
                lr=config['lr'],
                batch_size=config['batch_size']
            )
            
            resultados_cv.append({
                'config_idx': idx,
                'config': config,
                'cv_scores': avg_scores,
                'fold_scores': fold_scores,
                'original_f1': resultado_original['f1']
            })
            
            print(f"      ✅ CV F1: {avg_scores['f1']:.4f} ± {avg_scores['std_f1']:.4f}")
            print(f"      ✅ CV Precision: {avg_scores['precision']:.4f}")
            print(f"      ✅ CV Recall: {avg_scores['recall']:.4f}")
        
        # Guardar resultados de validación cruzada
        with open(f'resultados/validacion_cruzada_{idioma}.txt', 'w', encoding='utf-8') as f:
            f.write(f"RESULTADOS VALIDACIÓN CRUZADA ({k_folds}-folds) - {idioma.upper()}\n")
            f.write("="*80 + "\n\n")
            
            for i, res_cv in enumerate(resultados_cv):
                f.write(f"TOP {i+1} - Configuración {res_cv['config_idx']+1}:\n")
                f.write(f"  Parámetros: {res_cv['config']}\n")
                f.write(f"  F1 original: {res_cv['original_f1']:.4f}\n")
                f.write(f"  CV F1: {res_cv['cv_scores']['f1']:.4f} ± {res_cv['cv_scores']['std_f1']:.4f}\n")
                f.write(f"  CV Precision: {res_cv['cv_scores']['precision']:.4f}\n")
                f.write(f"  CV Recall: {res_cv['cv_scores']['recall']:.4f}\n")
                f.write(f"  CV Accuracy: {res_cv['cv_scores']['accuracy']:.4f}\n")
                f.write(f"  Tiempo promedio por fold: {res_cv['cv_scores']['tiempo_promedio']:.2f}s\n")
                f.write("-" * 60 + "\n")
        
        return resultados_cv
        
    except Exception as e:
        print(f"   ❌ Error en validación cruzada: {e}")
        return None

def main():
    print("🧠 PRÁCTICA 2: CLASIFICACIÓN DE HATE SPEECH CON MLP Y GPU")
    print("📊 Implementación desde cero con PyTorch y optimización GPU")
    print("="*80)
    
    # Verificar GPU
    verificar_gpu()
    
    # Crear directorios
    os.makedirs('resultados/graficas', exist_ok=True)
    
    # Ejecutar experimentos para ambos idiomas
    idiomas = ['es', 'en']
    todos_resultados = {}
    
    for idioma in idiomas:
        # Limpiar archivo de resultados
        with open(f'resultados/metricas_detalladas_{idioma}.txt', 'w', encoding='utf-8') as f:
            f.write(f"RESULTADOS DETALLADOS - {idioma.upper()}\n")
            f.write("="*80 + "\n\n")
        
        # Ejecutar experimento para el idioma
        resultados, tiempos = ejecutar_experimento_idioma(idioma)
        todos_resultados[idioma] = resultados
        
        if resultados:
            # Generar gráficas y tablas
            print(f"\n📈 Generando gráficas y tablas para {idioma.upper()}...")
            
            # Gráfica de TOP 5 configuraciones (requerido por PDF)
            graficar_top5_configuraciones(resultados, CONFIGURACIONES, idioma)
            
            # Gráficas adicionales
            graficar_perdidas(CONFIGURACIONES, resultados, top_n=5, idioma=idioma)
            graficar_metricas_comparativas(resultados, CONFIGURACIONES, idioma=idioma)
            graficar_evolucion_entrenamiento(resultados, CONFIGURACIONES, top_n=3, idioma=idioma)
            generar_tabla_resultados(resultados, CONFIGURACIONES, idioma=idioma)
            
            # Análisis final
            mejores_indices = np.argsort([r['f1'] for r in resultados])[-3:][::-1]
            
            print(f"\n🏆 MEJORES 3 CONFIGURACIONES - {idioma.upper()} (por F1-score):")
            for i, idx in enumerate(mejores_indices):
                res = resultados[idx]
                config = CONFIGURACIONES[idx]
                print(f"  {i+1}. Config {idx+1}: F1 = {res['f1']:.4f}, "
                      f"Precision = {res['precision']:.4f}, Recall = {res['recall']:.4f}")
                print(f"     Parámetros: {config}")
            
            print(f"\n⏱️  Tiempo total {idioma.upper()}: {sum(tiempos):.2f} segundos")
            print(f"⏱️  Tiempo promedio por configuración: {np.mean(tiempos):.2f} segundos")
    
    # Ejecutar validación cruzada para ambos idiomas
    print(f"\n{'='*80}")
    print("🎯 EJECUTANDO VALIDACIÓN CRUZADA PARA MEJORES CONFIGURACIONES")
    print(f"{'='*80}")
    
    for idioma in idiomas:
        if idioma in todos_resultados:
            resultados_cv = ejecutar_validacion_cruzada_completa(idioma, k_folds=5)
    
    # Generar análisis comparativo y gráfica ES vs EN
    if 'es' in todos_resultados and 'en' in todos_resultados:
        generar_analisis_comparativo(todos_resultados['es'], todos_resultados['en'])
        graficar_comparacion_es_en(todos_resultados['es'], todos_resultados['en'], CONFIGURACIONES)
    
    print(f"\n{'='*80}")
    print("🎉 EXPERIMENTO COMPLETADO EXITOSAMENTE")
    print(f"{'='*80}")
    print("📁 Resultados guardados en:")
    print("   - resultados/metricas_detalladas_[es|en].txt")
    print("   - resultados/tabla_resultados_[es|en].txt") 
    print("   - resultados/validacion_cruzada_[es|en].txt")
    print("   - resultados/analisis_comparativo.txt")
    print("   - resultados/graficas/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()