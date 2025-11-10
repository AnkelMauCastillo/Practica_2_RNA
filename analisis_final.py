# analisis_final.py
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def cargar_resultados_existentes():
    """Carga los resultados existentes de los archivos generados"""
    resultados_es = []
    resultados_en = []
    
    # Cargar configuración desde el archivo
    try:
        from configs_completas_gpu import CONFIGURACIONES
    except ImportError:
        print(" No se pudo cargar CONFIGURACIONES")
        return [], []
    
    # Cargar resultados desde los archivos de texto
    try:
        with open('resultados/metricas_detalladas_es.txt', 'r', encoding='utf-8') as f:
            contenido = f.read()
            # Parsear resultados (esto es un ejemplo simplificado)
            # En una implementación real, necesitarías un parseo más robusto
            pass
    except FileNotFoundError:
        print(" No se encontraron archivos de resultados")
    
    return resultados_es, resultados_en, CONFIGURACIONES

def graficar_error_vs_epocas_top5(resultados, configuraciones, idioma='es'):
    """
    Genera gráficas de error vs épocas para las 5 mejores configuraciones
    según lo requerido en el PDF
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
    Genera un análisis profundo de tendencias basado en los resultados experimentales
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
    best_es_neuronas = max(neuronas_results.items(), 
                          key=lambda x: x[1].get('es', 0))[0]
    best_en_neuronas = max(neuronas_results.items(), 
                          key=lambda x: x[1].get('en', 0))[0]
    
    analisis.append(f"\nÓPTIMO ESPAÑOL: {best_es_neuronas} neuronas")
    analisis.append(f"ÓPTIMO INGLÉS:  {best_en_neuronas} neuronas")
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
            if configs_lr_es[0]['train_losses']:
                conv_epochs_es = len(configs_lr_es[0]['train_losses'])
            else:
                conv_epochs_es = 'N/A'
            lr_results[lr] = {'es': (avg_es, conv_epochs_es)}
            
        if configs_lr_en:
            avg_en = np.mean([r['f1'] for r in configs_lr_en])
            # Analizar convergencia
            if configs_lr_en[0]['train_losses']:
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
    analisis.append("CONCLUSIONES PRINCIPALES")
    analisis.append("=" * 50)
    
    # Mejores configuraciones globales
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


def main():
    """Función principal para generar análisis final"""
    print(" GENERANDO ANÁLISIS FINAL Y GRÁFICAS")
    print("=" * 60)
    
    # Intentar cargar resultados existentes
    resultados_es, resultados_en, configuraciones = cargar_resultados_existentes()

    # Generar gráficas de error vs épocas
    print("\n GENERANDO GRÁFICAS DE ERROR VS ÉPOCAS")
    for idioma, resultados in [('es', resultados_es), ('en', resultados_en)]:
        graficar_error_vs_epocas_top5(resultados, configuraciones, idioma)
    
    # Generar análisis profundo
    print("\n GENERANDO ANÁLISIS PROFUNDO DE TENDENCIAS")
    generar_analisis_profundo_tendencias(resultados_es, resultados_en, configuraciones)
    
if __name__ == '__main__':
    main()