# main_gpu_completo.py
import json
import numpy as np
import os
import time
import torch
import matplotlib.pyplot as plt
from src.preprocesamiento import Preprocesador
from src.representaciones import crear_vectorizador
from src.mlp_gpu import MLP_GPU
from src.entrenamiento_gpu import entrenar_mlp_gpu_simple
from src.evaluacion_gpu import evaluar_modelo_gpu, validacion_cruzada_gpu
from src.visualizacion import (graficar_perdidas, graficar_metricas_comparativas, 
                              generar_tabla_resultados, graficar_evolucion_entrenamiento,
                              graficar_top5_configuraciones, graficar_comparacion_es_en)
from configs_completas_gpu import CONFIGURACIONES

# =============================================================================
# ANÁLISIS Y GRÁFICAS
# =============================================================================

def graficar_error_vs_epocas_top5(resultados, configuraciones, idioma='es'):
    """
    Genera gráficas de error vs épocas para las 5 mejores configuraciones
    """
    print(f" Generando gráficas de error vs épocas para {idioma.upper()}...")
    
    # Ordenar por F1-score y tomar las 5 mejores
    indices_mejores = np.argsort([r['f1'] for r in resultados])[-5:][::-1]
    
    # Crear figura principal
    fig = plt.figure(figsize=(16, 12))
    
    for i, idx in enumerate(indices_mejores):
        config = configuraciones[idx]
        resultado = resultados[idx]
        
        # Crear subplot para esta configuración
        plt.subplot(3, 2, i+1)
        
        # Verificar que tenemos datos de pérdida
        if (len(resultado['train_losses']) > 0 and len(resultado['test_losses']) > 0 and
            len(resultado['train_losses']) == len(resultado['test_losses'])):
            
            epochs = range(1, len(resultado['train_losses']) + 1)
            
            # Graficar curvas de pérdida
            plt.plot(epochs, resultado['train_losses'], 
                    label='Pérdida Entrenamiento', linewidth=2.5, color='blue', alpha=0.8)
            plt.plot(epochs, resultado['test_losses'], 
                    label='Pérdida Validación', linewidth=2.5, color='red', alpha=0.8)
            
            # Configurar el gráfico
            plt.title(f'Config {idx+1} - F1: {resultado["f1"]:.4f}', 
                     fontsize=12, fontweight='bold', pad=10)
            plt.xlabel('Épocas', fontsize=10)
            plt.ylabel('Error Cuadrático Medio (MSE)', fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            
            # Añadir información de configuración
            config_info = (
                f"Neuronas: {config['neuronas_ocultas']}\n"
                f"LR: {config['lr']}, Batch: {config['batch_size']}\n"
                f"N-grams: {config['ngramas']}\n"
                f"Pesado: {config['pesado_terminos']}\n"
                f"Inicial: {config['inicializacion']}"
            )
            
            plt.text(0.02, 0.98, config_info, transform=plt.gca().transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
            
            # Resaltar época de mejor pérdida de validación
            best_epoch = np.argmin(resultado['test_losses'])
            best_loss = resultado['test_losses'][best_epoch]
            plt.axvline(x=best_epoch+1, color='red', linestyle='--', alpha=0.5)
            plt.plot(best_epoch+1, best_loss, 'ro', markersize=6)
            plt.text(best_epoch+1, best_loss, f'  Mejor: {best_loss:.4f}', 
                    fontsize=8, verticalalignment='bottom')
            
        else:
            # Si no hay datos de pérdida
            plt.text(0.5, 0.5, 'Datos de pérdida no disponibles', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'Config {idx+1} - F1: {resultado["f1"]:.3f}', fontsize=12)
    
    # Título principal
    plt.suptitle(f'TOP 5 CONFIGURACIONES - Error vs Épocas ({idioma.upper()})\n'
                f'Comportamiento del Error Cuadrático Medio durante el Entrenamiento', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Guardar gráfica
    os.makedirs('resultados/graficas', exist_ok=True)
    filename = f'resultados/graficas/error_vs_epocas_top5_{idioma}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   Gráfica de error vs épocas guardada: {filename}")
    return filename

def generar_analisis_profundo_tendencias(resultados_es, resultados_en, configuraciones):
    """
    Genera un análisis profundo de tendencias basado en los resultados 
    """
    print(" Generando análisis profundo de tendencias...")
    
    analisis = []
    
    # 1. ANÁLISIS DE INICIALIZACIÓN
    analisis.append("ANÁLISIS DE INICIALIZACIÓN DE PESOS")
    analisis.append("=" * 50)
    
    xavier_es = [r for i, r in enumerate(resultados_es) 
                if configuraciones[i]['inicializacion'] == 'xavier' 
                and configuraciones[i]['ngramas'] == (1,1)]
    normal_es = [r for i, r in enumerate(resultados_es) 
                if configuraciones[i]['inicializacion'] == 'normal' 
                and configuraciones[i]['ngramas'] == (1,1)]
    
    xavier_en = [r for i, r in enumerate(resultados_en) 
                if configuraciones[i]['inicializacion'] == 'xavier' 
                and configuraciones[i]['ngramas'] == (1,1)]
    normal_en = [r for i, r in enumerate(resultados_en) 
                if configuraciones[i]['inicializacion'] == 'normal' 
                and configuraciones[i]['ngramas'] == (1,1)]
    
    if xavier_es and normal_es:
        avg_xavier_es = np.mean([r['f1'] for r in xavier_es])
        avg_normal_es = np.mean([r['f1'] for r in normal_es])
        analisis.append(f"ESPAÑOL - Xavier: {avg_xavier_es:.4f}, Normal: {avg_normal_es:.4f}")
        analisis.append(f"   Mejor inicialización: {'XAVIER' if avg_xavier_es > avg_normal_es else 'NORMAL'}")
        analisis.append(f"   Diferencia: {abs(avg_xavier_es - avg_normal_es):.4f}")
    
    if xavier_en and normal_en:
        avg_xavier_en = np.mean([r['f1'] for r in xavier_en])
        avg_normal_en = np.mean([r['f1'] for r in normal_en])
        analisis.append(f"INGLÉS  - Xavier: {avg_xavier_en:.4f}, Normal: {avg_normal_en:.4f}")
        analisis.append(f"   Mejor inicialización: {'XAVIER' if avg_xavier_en > avg_normal_en else 'NORMAL'}")
        analisis.append(f"   Diferencia: {abs(avg_xavier_en - avg_normal_en):.4f}")
    
    analisis.append("")
    
    # 2. ANÁLISIS DE NEURONAS OCULTAS
    analisis.append("ANÁLISIS DE NEURONAS EN CAPA OCULTA")
    analisis.append("=" * 50)
    
    neuronas_results = {}
    for neuronas in [64, 128, 256, 512, 1024]:
        configs_neuronas_es = [r for i, r in enumerate(resultados_es) 
                              if configuraciones[i]['neuronas_ocultas'] == neuronas 
                              and configuraciones[i]['ngramas'] == (1,1)]
        configs_neuronas_en = [r for i, r in enumerate(resultados_en) 
                              if configuraciones[i]['neuronas_ocultas'] == neuronas 
                              and configuraciones[i]['ngramas'] == (1,1)]
        
        if configs_neuronas_es:
            avg_es = np.mean([r['f1'] for r in configs_neuronas_es])
            neuronas_results[neuronas] = {'es': avg_es}
        if configs_neuronas_en:
            avg_en = np.mean([r['f1'] for r in configs_neuronas_en])
            neuronas_results.setdefault(neuronas, {})['en'] = avg_en
    
    for neuronas, scores in sorted(neuronas_results.items()):
        es_score = scores.get('es', 0)
        en_score = scores.get('en', 0)
        analisis.append(f"{neuronas:4d} neuronas - ES: {es_score:.4f}, EN: {en_score:.4f}")
    
    # Encontrar óptimo por idioma
    if neuronas_results:
        best_es_neuronas = max(neuronas_results.items(), 
                              key=lambda x: x[1].get('es', 0))[0]
        best_en_neuronas = max(neuronas_results.items(), 
                              key=lambda x: x[1].get('en', 0))[0]
        
        analisis.append(f"\nÓPTIMO ESPAÑOL: {best_es_neuronas} neuronas")
        analisis.append(f"ÓPTIMO INGLÉS:  {best_en_neuronas} neuronas")
    else:
        analisis.append("\nNo hay datos suficientes para análisis de neuronas")
    
    analisis.append("")
    
    # 3. ANÁLISIS DE LEARNING RATE
    analisis.append("ANÁLISIS DE TASA DE APRENDIZAJE (Learning Rate)")
    analisis.append("=" * 50)
    
    lr_results = {}
    for lr in [0.01, 0.1, 0.5]:
        configs_lr_es = [r for i, r in enumerate(resultados_es) 
                        if configuraciones[i]['lr'] == lr 
                        and configuraciones[i]['neuronas_ocultas'] == 128 
                        and configuraciones[i]['ngramas'] == (1,1)]
        configs_lr_en = [r for i, r in enumerate(resultados_en) 
                        if configuraciones[i]['lr'] == lr 
                        and configuraciones[i]['neuronas_ocultas'] == 128 
                        and configuraciones[i]['ngramas'] == (1,1)]
        
        if configs_lr_es:
            avg_es = np.mean([r['f1'] for r in configs_lr_es])
            # Analizar convergencia
            if configs_lr_es and configs_lr_es[0]['train_losses']:
                conv_epochs_es = len(configs_lr_es[0]['train_losses'])
            else:
                conv_epochs_es = 'N/A'
            lr_results[lr] = {'es': (avg_es, conv_epochs_es)}
            
        if configs_lr_en:
            avg_en = np.mean([r['f1'] for r in configs_lr_en])
            # Analizar convergencia
            if configs_lr_en and configs_lr_en[0]['train_losses']:
                conv_epochs_en = len(configs_lr_en[0]['train_losses'])
            else:
                conv_epochs_en = 'N/A'
            lr_results.setdefault(lr, {})['en'] = (avg_en, conv_epochs_en)
    
    for lr, scores in sorted(lr_results.items()):
        es_info = scores.get('es', (0, 'N/A'))
        en_info = scores.get('en', (0, 'N/A'))
        analisis.append(f"LR {lr:.2f} - ES: F1={es_info[0]:.4f} (épocas: {es_info[1]}), "
                      f"EN: F1={en_info[0]:.4f} (épocas: {en_info[1]})")
    
    analisis.append("")
    
    # 4. ANÁLISIS DE PREPROCESAMIENTO
    analisis.append("ANÁLISIS DE PREPROCESAMIENTO")
    analisis.append("=" * 50)
    
    preproc_types = ['normalizar', 'normalizar_sin_stopwords', 'normalizar_sin_stopwords_stemming']
    preproc_names = ['Normal', 'Sin StopWords', 'Sin SW + Stemming']
    
    for preproc, name in zip(preproc_types, preproc_names):
        configs_preproc_es = [r for i, r in enumerate(resultados_es) 
                             if configuraciones[i]['preprocesamiento'] == preproc 
                             and configuraciones[i]['neuronas_ocultas'] == 128 
                             and configuraciones[i]['ngramas'] == (1,1)]
        configs_preproc_en = [r for i, r in enumerate(resultados_en) 
                             if configuraciones[i]['preprocesamiento'] == preproc 
                             and configuraciones[i]['neuronas_ocultas'] == 128 
                             and configuraciones[i]['ngramas'] == (1,1)]
        
        if configs_preproc_es:
            avg_es = np.mean([r['f1'] for r in configs_preproc_es])
            analisis.append(f"{name:15} - ES: {avg_es:.4f}")
        if configs_preproc_en:
            avg_en = np.mean([r['f1'] for r in configs_preproc_en])
            analisis.append(f"{name:15} - EN: {avg_en:.4f}")
    
    analisis.append("")
    
    # 5. ANÁLISIS DE N-GRAMAS
    analisis.append("ANÁLISIS DE N-GRAMAS")
    analisis.append("=" * 50)
    
    ngram_types = [(1,1), (2,2), (1,2)]
    ngram_names = ['Unigramas', 'Bigramas', 'Uni+Bigramas']
    
    for ngram, name in zip(ngram_types, ngram_names):
        configs_ngram_es = [r for i, r in enumerate(resultados_es) 
                           if configuraciones[i]['ngramas'] == ngram 
                           and configuraciones[i]['neuronas_ocultas'] == 128]
        configs_ngram_en = [r for i, r in enumerate(resultados_en) 
                           if configuraciones[i]['ngramas'] == ngram 
                           and configuraciones[i]['neuronas_ocultas'] == 128]
        
        if configs_ngram_es:
            avg_es = np.mean([r['f1'] for r in configs_ngram_es])
            analisis.append(f"{name:15} - ES: {avg_es:.4f}")
        if configs_ngram_en:
            avg_en = np.mean([r['f1'] for r in configs_ngram_en])
            analisis.append(f"{name:15} - EN: {avg_en:.4f}")
    
    analisis.append("")
    
    # 6. ANÁLISIS DE TF vs TF-IDF
    analisis.append("ANÁLISIS DE PESADO DE TÉRMINOS")
    analisis.append("=" * 50)
    
    for pesado in ['tf', 'tfidf']:
        configs_pesado_es = [r for i, r in enumerate(resultados_es) 
                            if configuraciones[i]['pesado_terminos'] == pesado 
                            and configuraciones[i]['neuronas_ocultas'] == 128 
                            and configuraciones[i]['ngramas'] == (1,1)]
        configs_pesado_en = [r for i, r in enumerate(resultados_en) 
                            if configuraciones[i]['pesado_terminos'] == pesado 
                            and configuraciones[i]['neuronas_ocultas'] == 128 
                            and configuraciones[i]['ngramas'] == (1,1)]
        
        if configs_pesado_es:
            avg_es = np.mean([r['f1'] for r in configs_pesado_es])
            analisis.append(f"{pesado.upper():6} - ES: {avg_es:.4f}")
        if configs_pesado_en:
            avg_en = np.mean([r['f1'] for r in configs_pesado_en])
            analisis.append(f"{pesado.upper():6} - EN: {avg_en:.4f}")
    
    analisis.append("")
    
    # 7. CONCLUSIONES GENERALES
    analisis.append("CONCLUSIONES GENERALES")
    analisis.append("=" * 50)
    
    # Mejores configuraciones globales
    if resultados_es and resultados_en:
        mejor_es_idx = np.argmax([r['f1'] for r in resultados_es])
        mejor_en_idx = np.argmax([r['f1'] for r in resultados_en])
        
        mejor_es = resultados_es[mejor_es_idx]
        mejor_en = resultados_en[mejor_en_idx]
        
        analisis.append(f"MEJOR CONFIGURACIÓN ESPAÑOL (F1: {mejor_es['f1']:.4f}):")
        analisis.append(f"  {configuraciones[mejor_es_idx]}")
        analisis.append(f"MEJOR CONFIGURACIÓN INGLÉS (F1: {mejor_en['f1']:.4f}):")
        analisis.append(f"  {configuraciones[mejor_en_idx]}")
        analisis.append("")
        
        # Diferencias entre idiomas
        diff_f1 = mejor_es['f1'] - mejor_en['f1']
        analisis.append(f"DIFERENCIA ENTRE IDIOMAS: {diff_f1:.4f} (ES mejor por {diff_f1*100:.1f}%)")
    else:
        analisis.append("No hay resultados suficientes para conclusiones")
    
    analisis.append("")
    
    # Guardar análisis
    os.makedirs('resultados', exist_ok=True)
    with open('resultados/analisis_profundo_tendencias.txt', 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS PROFUNDO DE TENDENCIAS EXPERIMENTALES\n")
        f.write("=" * 60 + "\n\n")
        for linea in analisis:
            f.write(linea + "\n")
    
    print("   Análisis profundo guardado: resultados/analisis_profundo_tendencias.txt")
    return analisis

# =============================================================================
# FUNCIONES PRINCIPALES DE VALIDACIÓN CRUZADA
# =============================================================================

def ejecutar_validacion_cruzada_completa(idioma='es', k_folds=5):
    """Ejecuta validación cruzada para las 3 mejores configuraciones"""
    print(f"\n INICIANDO VALIDACIÓN CRUZADA ({k_folds}-folds) - {idioma.upper()}")
    
    try:
        # Cargar todos los datos
        X_todos, y_todos = cargar_datos(f'data/hateval_{idioma}_all.json')
        print(f"    Datos cargados: {len(X_todos)} ejemplos")
        print(f"    Distribución de clases: {np.bincount(y_todos)}")
        
        # Identificar las 3 mejores configuraciones basadas en resultados previos
        if idioma == 'es':
            mejores_config_indices = [8, 0, 1]  # Ajustado: Config 9, 1, 2
            mejores_configs = [
                CONFIGURACIONES[8],  # Config 9: (1,2) n-grams
                CONFIGURACIONES[0],  # Config 1: 64 neuronas
                CONFIGURACIONES[1]   # Config 2: 128 neuronas
            ]
        else:  # 'en'
            mejores_config_indices = [8, 0, 4]  # Ajustado: Config 9, 1, 5
            mejores_configs = [
                CONFIGURACIONES[8],  # Config 9: (1,2) n-grams
                CONFIGURACIONES[0],  # Config 1: 64 neuronas  
                CONFIGURACIONES[4]   # Config 5: 1024 neuronas
            ]
        
        resultados_cv = {}
        
        for i, (config_idx, config) in enumerate(zip(mejores_config_indices, mejores_configs)):
            print(f"\n    Configuración {i+1}/3 (Original: Config {config_idx+1}):")
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
            
            print(f"       Datos vectorizados: {X_vec.shape}")
            
            # Ejecutar validación cruzada
            cv_start = time.time()
            avg_scores, fold_scores = validacion_cruzada_gpu(
                config, X_vec, y_vec, entrenar_mlp_gpu_simple, k_folds=k_folds,
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

            print(f"   Resultados CV:")
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
        print(f"   Error en validación cruzada: {e}")
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
    if not resultados_es or not resultados_en:
        print(" No hay resultados suficientes para análisis comparativo")
        return
    
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
        print(f" GPUs disponibles: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Configurar GPU por defecto
        torch.cuda.set_device(0)
        print(f"   Usando GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print(" No se encontró GPU compatible con CUDA")
        return False

def cargar_datos(archivo):
    """Carga datos desde archivo JSON o JSONL"""
    datos = []
    try:
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
    except FileNotFoundError:
        print(f" Archivo no encontrado: {archivo}")
        return [], []
    except Exception as e:
        print(f" Error cargando {archivo}: {e}")
        return [], []
    
    textos = [d.get('text', '') for d in datos]
    etiquetas = [d.get('klass', 0) for d in datos]
    return textos, etiquetas

def aplicar_preprocesamiento(config, textos, idioma='es'):
    """Aplica el preprocesamiento confoeme a la configuración"""
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
    print(f" INICIANDO EXPERIMENTO PARA IDIOMA: {idioma.upper()}")
    print(f"{'='*80}")
    
    # Cargar datos
    print(" Cargando datos...")
    try:
        X_entrenamiento, y_entrenamiento = cargar_datos(f'data/hateval_{idioma}_train.json')
        X_prueba, y_prueba = cargar_datos(f'data/hateval_{idioma}_test.json')
        
        if len(X_entrenamiento) == 0 or len(X_prueba) == 0:
            print(f"    No se pudieron cargar datos para {idioma}")
            return [], []
        
        print(f"    Datos cargados:")
        print(f"      - Entrenamiento: {len(X_entrenamiento)} ejemplos")
        print(f"      - Prueba: {len(X_prueba)} ejemplos")
        print(f"      - Distribución clases (train): {np.bincount(y_entrenamiento)}")
        print(f"      - Distribución clases (test): {np.bincount(y_prueba)}")
        
    except Exception as e:
        print(f"   Error cargando datos: {e}")
        return [], []

    # Probar todas las configuraciones
    resultados = []
    tiempos_ejecucion = []
    
    print(f"\n Probando {len(CONFIGURACIONES)} configuraciones para {idioma.upper()}...")
    
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
            print("   Preprocesando textos...")
            X_ent_limpio = aplicar_preprocesamiento(config, X_entrenamiento, idioma)
            X_prueba_limpio = aplicar_preprocesamiento(config, X_prueba, idioma)

            # Vectorización
            print("   Vectorizando textos...")
            vectorizador = crear_vectorizador(
                tipo=config['pesado_terminos'],
                ngram_range=config['ngramas']
            )
            X_ent_vec = vectorizador.fit_transform(X_ent_limpio).toarray()
            X_prueba_vec = vectorizador.transform(X_prueba_limpio).toarray()

            print(f"      Dimensionalidad: {X_ent_vec.shape[1]} features")
            print(f"      Memoria: {(X_ent_vec.nbytes + X_prueba_vec.nbytes) / 1024**2:.2f} MB")

            # Crear y entrenar modelo en GPU
            modelo = MLP_GPU(
                input_size=X_ent_vec.shape[1],
                hidden_size=config['neuronas_ocultas'],
                inicializacion=config['inicializacion']
            )

            y_ent = np.array(y_entrenamiento).reshape(-1, 1)
            y_pru = np.array(y_prueba).reshape(-1, 1)

            print("   Entrenando modelo en GPU...")
            metricas_entrenamiento = entrenar_mlp_gpu_simple(
                modelo, X_ent_vec, y_ent, X_prueba_vec, y_pru,
                epochs=config['epochs'], batch_size=config['batch_size'], lr=config['lr']
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
            
            print(f"\n   RESULTADOS:")
            print(f"      Precision: {resultados_evaluacion['precision']:.4f}")
            print(f"      Recall: {resultados_evaluacion['recall']:.4f}")
            print(f"      F1-score: {resultados_evaluacion['f1']:.4f}")
            print(f"      Accuracy: {resultados_evaluacion['accuracy']:.4f}")
            print(f"      Tiempo: {tiempo_ejecucion:.2f} segundos")

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
            print(f"   Error en configuración {config_idx + 1}: {e}")
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
            tiempos_ejecucion.append(0)

    return resultados, tiempos_ejecucion

# =============================================================================
# MAIN CORREGIDO
# =============================================================================

def main():
    print(" PRÁCTICA 2: CLASIFICACIÓN DE HATE SPEECH CON MLP Y GPU")
    
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
        print(f"\n  INICIANDO EXPERIMENTO PARA {idioma.upper()}")
        resultados, tiempos = ejecutar_experimento_idioma(idioma)
        
        if resultados and len(resultados) > 0:
            todos_resultados[idioma] = resultados
            
            # Generar gráficas y tablas
            print(f"\n Generando gráficas y tablas para {idioma.upper()}...")
            
            # Gráfica de error vs épocas para TOP 5 configuraciones 
            graficar_error_vs_epocas_top5(resultados, CONFIGURACIONES, idioma)
            
            # Gráficas adicionales existentes
            graficar_top5_configuraciones(resultados, CONFIGURACIONES, idioma)
            graficar_perdidas(CONFIGURACIONES, resultados, top_n=5, idioma=idioma)
            graficar_metricas_comparativas(resultados, CONFIGURACIONES, idioma=idioma)
            graficar_evolucion_entrenamiento(resultados, CONFIGURACIONES, top_n=3, idioma=idioma)
            generar_tabla_resultados(resultados, CONFIGURACIONES, idioma=idioma)
            
            # Análisis final
            mejores_indices = np.argsort([r['f1'] for r in resultados])[-3:][::-1]
            
            print(f"\n MEJORES 3 CONFIGURACIONES - {idioma.upper()} (por F1-score):")
            for i, idx in enumerate(mejores_indices):
                res = resultados[idx]
                config = CONFIGURACIONES[idx]
                print(f"  {i+1}. Config {idx+1}: F1 = {res['f1']:.4f}, "
                      f"Precision = {res['precision']:.4f}, Recall = {res['recall']:.4f}")
                print(f"     Parámetros: {config}")
            
            print(f"\n  Tiempo total {idioma.upper()}: {sum(tiempos):.2f} segundos")
            print(f"  Tiempo promedio por configuración: {np.mean(tiempos):.2f} segundos")
        else:
            print(f" No se obtuvieron resultados para {idioma}")
    
    # Ejecutar validación cruzada para ambos idiomas
    print(f"\n{'='*80}")
    print(" EJECUTANDO VALIDACIÓN CRUZADA PARA MEJORES CONFIGURACIONES")
    print(f"{'='*80}")
    
    for idioma in idiomas:
        if idioma in todos_resultados and len(todos_resultados[idioma]) > 0:
            resultados_cv = ejecutar_validacion_cruzada_completa(idioma, k_folds=5)
    
    # Generar análisis comparativo y gráfica ES vs EN
    if 'es' in todos_resultados and 'en' in todos_resultados:
        if len(todos_resultados['es']) > 0 and len(todos_resultados['en']) > 0:
            print(f"\n{'='*80}")
            print(" GENERANDO ANÁLISIS PROFUNDO DE TENDENCIAS")
            print(f"{'='*80}")
            
            # Análisis profundo de tendencias
            generar_analisis_profundo_tendencias(todos_resultados['es'], todos_resultados['en'], CONFIGURACIONES)
            
            # Análisis comparativo existente
            generar_analisis_comparativo(todos_resultados['es'], todos_resultados['en'])
            graficar_comparacion_es_en(todos_resultados['es'], todos_resultados['en'], CONFIGURACIONES)
    
    print(f"\n{'='*80}")
    print(" EXPERIMENTO COMPLETADO EXITOSAMENTE")
    print(f"{'='*80}")
    

if __name__ == '__main__':
    main()