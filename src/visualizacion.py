import matplotlib.pyplot as plt
import numpy as np
import os

def graficar_top5_configuraciones(resultados, configuraciones, idioma='es'):
    """Grafica las curvas de pérdida para las 5 mejores configuraciones (requerido por PDF)"""
    
    # Ordenar por F1-score y tomar las 5 mejores
    indices_mejores = np.argsort([r['f1'] for r in resultados])[-5:][::-1]
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices_mejores):
        config = configuraciones[idx]
        resultado = resultados[idx]
        
        plt.subplot(2, 3, i+1)
        
        # Verificar que tenemos datos de pérdida
        if len(resultado['train_losses']) > 0 and len(resultado['test_losses']) > 0:
            epochs = range(len(resultado['train_losses']))
            
            # Curvas de pérdida
            plt.plot(epochs, resultado['train_losses'], 
                    label='Pérdida Entrenamiento', linewidth=2, alpha=0.8, color='blue')
            plt.plot(epochs, resultado['test_losses'], 
                    label='Pérdida Prueba', linewidth=2, alpha=0.8, color='red')
            
            plt.title(f'Config {idx+1}\nF1: {resultado["f1"]:.3f}', fontsize=12, fontweight='bold')
            plt.xlabel('Épocas')
            plt.ylabel('Pérdida (MSE)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Información de configuración en texto
            config_text = (f"Neuronas: {config['neuronas_ocultas']}\n"
                          f"LR: {config['lr']}, Batch: {config['batch_size']}\n"
                          f"N-grams: {config['ngramas']}\n"
                          f"Pesado: {config['pesado_terminos']}")
            
            plt.text(0.02, 0.98, config_text, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            plt.text(0.5, 0.5, 'Sin datos de pérdida', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'Config {idx+1}\nF1: {resultado["f1"]:.3f}', fontsize=12)
    
    # Ocultar el sexto subplot si solo hay 5 configuraciones
    if len(indices_mejores) < 6:
        plt.delaxes(plt.subplot(2, 3, 6))
    
    plt.suptitle(f'TOP 5 CONFIGURACIONES - Curvas de Pérdida vs Épocas ({idioma.upper()})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'resultados/graficas/top5_perdidas_{idioma}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Gráfica TOP 5 configuraciones guardada: top5_perdidas_{idioma}.png")

def graficar_comparacion_es_en(resultados_es, resultados_en, configuraciones):
    """Grafica comparativa directa entre español e inglés"""
    
    # Métricas promedio por tipo de configuración
    metricas_es = {
        'neuronas': [],
        'lr': [],
        'ngramas': [],
        'preprocesamiento': []
    }
    
    metricas_en = {
        'neuronas': [],
        'lr': [], 
        'ngramas': [],
        'preprocesamiento': []
    }
    
    # Agrupar configuraciones similares
    for config, res_es, res_en in zip(configuraciones, resultados_es, resultados_en):
        # Por neuronas (solo configs base)
        if (config['ngramas'] == (1,1) and config['preprocesamiento'] == 'normalizar' and 
            config['inicializacion'] == 'xavier' and config['pesado_terminos'] == 'tf' and
            config['lr'] == 0.01 and config['batch_size'] == 128):
            metricas_es['neuronas'].append((config['neuronas_ocultas'], res_es['f1']))
            metricas_en['neuronas'].append((config['neuronas_ocultas'], res_en['f1']))
    
    # Graficar comparación de neuronas
    if metricas_es['neuronas']:
        neuronas_es = sorted(metricas_es['neuronas'], key=lambda x: x[0])
        neuronas_en = sorted(metricas_en['neuronas'], key=lambda x: x[0])
        
        plt.figure(figsize=(10, 6))
        x = [n[0] for n in neuronas_es]
        y_es = [n[1] for n in neuronas_es]
        y_en = [n[1] for n in neuronas_en]
        
        plt.plot(x, y_es, 'o-', label='Español', linewidth=2, markersize=8)
        plt.plot(x, y_en, 's-', label='Inglés', linewidth=2, markersize=8)
        
        plt.xlabel('Neuronas en Capa Oculta')
        plt.ylabel('F1-score')
        plt.title('Comparación Español vs Inglés: Neuronas Ocultas')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(x)
        
        plt.tight_layout()
        plt.savefig('resultados/graficas/comparacion_es_en_neuronas.png', dpi=300, bbox_inches='tight')
        plt.close()

def graficar_perdidas(configuraciones, resultados, top_n=5, idioma='es'):
    """Grafica las curvas de pérdida para las mejores configuraciones"""
    
    # Ordenar por F1-score y tomar las mejores
    indices_mejores = np.argsort([r['f1'] for r in resultados])[-top_n:][::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices_mejores):
        if i >= len(axes):
            break
            
        config = configuraciones[idx]
        resultado = resultados[idx]
        
        # Verificar que tenemos datos de pérdida
        if len(resultado['train_losses']) > 0 and len(resultado['test_losses']) > 0:
            epochs = range(len(resultado['train_losses']))
            
            axes[i].plot(epochs, resultado['train_losses'], label='Pérdida Entrenamiento', linewidth=2, alpha=0.8)
            axes[i].plot(epochs, resultado['test_losses'], label='Pérdida Prueba', linewidth=2, alpha=0.8)
            axes[i].set_title(f'Config {idx+1}\nF1: {resultado["f1"]:.3f}', fontsize=10)
            axes[i].set_xlabel('Épocas')
            axes[i].set_ylabel('Pérdida (MSE)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'Sin datos de pérdida', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[i].transAxes)
            axes[i].set_title(f'Config {idx+1}\nF1: {resultado["f1"]:.3f}', fontsize=10)
    
    # Ocultar ejes vacíos
    for i in range(len(indices_mejores), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Curvas de Pérdida - Top {len(indices_mejores)} Configuraciones ({idioma.upper()})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'resultados/graficas/perdidas_mejores_{idioma}.png', dpi=300, bbox_inches='tight')
    plt.close()


def graficar_metricas_comparativas(resultados, configuraciones, idioma='es'):
    """Grafica comparativa de métricas para todas las configuraciones"""
    f1_scores = [r['f1'] for r in resultados]
    precision_scores = [r['precision'] for r in resultados]
    recall_scores = [r['recall'] for r in resultados]
    
    x = np.arange(len(resultados))
    width = 0.25
    
    plt.figure(figsize=(16, 6))
    
    plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.7, color='skyblue')
    plt.bar(x, recall_scores, width, label='Recall', alpha=0.7, color='lightcoral')
    plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.7, color='lightgreen')
    
    plt.xlabel('Configuraciones')
    plt.ylabel('Puntuación')
    plt.title(f'Comparación de Métricas por Configuración ({idioma.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(x, [f'C{i+1}' for i in range(len(resultados))], rotation=45)
    plt.ylim(0, 1)
    
    # Añadir valores en las barras
    for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
        plt.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        plt.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        plt.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'resultados/graficas/comparacion_metricas_{idioma}.png', dpi=300, bbox_inches='tight')
    plt.close()


def generar_tabla_resultados(resultados, configuraciones, idioma='es'):
    """Genera una tabla completa con los resultados"""
    
    # Guardar tabla en archivo
    with open(f'resultados/tabla_resultados_{idioma}.txt', 'w', encoding='utf-8') as f:
        f.write(f"TABLA COMPLETA DE RESULTADOS - {idioma.upper()}\n")
        f.write("="*120 + "\n")
        f.write(f"{'Config':^6} {'Neuronas':^8} {'Inicial':^10} {'Pesado':^8} {'Ngramas':^10} {'Preproc':^25} {'LR':^6} {'Batch':^6} {'F1':^8} {'Precision':^10} {'Recall':^8} {'Tiempo':^10}\n")
        f.write("-"*120 + "\n")
        
        for i, (config, res) in enumerate(zip(configuraciones, resultados)):
            preproc_map = {
                'normalizar': 'Normal',
                'normalizar_sin_stopwords': 'Sin StopWords',
                'normalizar_sin_stopwords_stemming': 'Sin SW + Stem'
            }
            preproc_str = preproc_map.get(config['preprocesamiento'], config['preprocesamiento'])
            
            f.write(f"{i+1:^6} {config['neuronas_ocultas']:^8} {config['inicializacion']:^10} "
                   f"{config['pesado_terminos']:^8} {str(config['ngramas']):^10} {preproc_str:^25} "
                   f"{config['lr']:^6} {config['batch_size']:^6} {res['f1']:^8.3f} {res['precision']:^10.3f} {res['recall']:^8.3f} {res['tiempo_ejecucion']:^10.1f}\n")
    
    # También imprimir en consola
    print(f"\n{'='*120}")
    print(f"TABLA COMPLETA DE RESULTADOS - {idioma.upper()}")
    print("="*120)
    print(f"{'Config':^6} {'Neuronas':^8} {'Inicial':^10} {'Pesado':^8} {'Ngramas':^10} {'Preproc':^25} {'LR':^6} {'Batch':^6} {'F1':^8} {'Precision':^10} {'Recall':^8} {'Tiempo':^10}")
    print("-"*120)
    
    for i, (config, res) in enumerate(zip(configuraciones, resultados)):
        preproc_map = {
            'normalizar': 'Normal',
            'normalizar_sin_stopwords': 'Sin StopWords',
            'normalizar_sin_stopwords_stemming': 'Sin SW + Stem'
        }
        preproc_str = preproc_map.get(config['preprocesamiento'], config['preprocesamiento'])
        
        print(f"{i+1:^6} {config['neuronas_ocultas']:^8} {config['inicializacion']:^10} "
              f"{config['pesado_terminos']:^8} {str(config['ngramas']):^10} {preproc_str:^25} "
              f"{config['lr']:^6} {config['batch_size']:^6} {res['f1']:^8.3f} {res['precision']:^10.3f} {res['recall']:^8.3f} {res['tiempo_ejecucion']:^10.1f}")

def graficar_evolucion_entrenamiento(resultados, configuraciones, top_n=3, idioma='es'):
    """Grafica la evolución del accuracy durante el entrenamiento"""
    indices_mejores = np.argsort([r['f1'] for r in resultados])[-top_n:][::-1]
    
    plt.figure(figsize=(12, 8))
    
    for i, idx in enumerate(indices_mejores):
        config = configuraciones[idx]
        resultado = resultados[idx]
        
        if ('train_accuracies' in resultado and 'test_accuracies' in resultado and
            len(resultado['train_accuracies']) > 0 and len(resultado['test_accuracies']) > 0):
            epochs = range(len(resultado['train_accuracies']))
            
            plt.subplot(2, 2, i+1)
            plt.plot(epochs, resultado['train_accuracies'], label='Accuracy Entrenamiento', linewidth=2)
            plt.plot(epochs, resultado['test_accuracies'], label='Accuracy Prueba', linewidth=2)
            plt.title(f'Config {idx+1} - F1: {resultado["f1"]:.3f}\nNeuronas: {config["neuronas_ocultas"]}, LR: {config["lr"]}')
            plt.xlabel('Épocas')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
        else:
            plt.subplot(2, 2, i+1)
            plt.text(0.5, 0.5, 'Sin datos de accuracy', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.title(f'Config {idx+1} - F1: {resultado["f1"]:.3f}')
    
    plt.suptitle(f'Evolución del Accuracy durante el Entrenamiento - {idioma.upper()}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'resultados/graficas/evolucion_accuracy_{idioma}.png', dpi=300, bbox_inches='tight')
    plt.close()

def graficar_evolucion_entrenamiento(resultados, configuraciones, top_n=3, idioma='es'):
    """Grafica la evolución del accuracy durante el entrenamiento"""
    indices_mejores = np.argsort([r['f1'] for r in resultados])[-top_n:][::-1]
    
    plt.figure(figsize=(12, 8))
    
    for i, idx in enumerate(indices_mejores):
        config = configuraciones[idx]
        resultado = resultados[idx]
        
        if 'train_accuracies' in resultado and 'test_accuracies' in resultado:
            epochs = range(len(resultado['train_accuracies']))
            
            plt.subplot(2, 2, i+1)
            plt.plot(epochs, resultado['train_accuracies'], label='Accuracy Entrenamiento', linewidth=2)
            plt.plot(epochs, resultado['test_accuracies'], label='Accuracy Prueba', linewidth=2)
            plt.title(f'Config {idx+1} - F1: {resultado["f1"]:.3f}\nNeuronas: {config["neuronas_ocultas"]}, LR: {config["lr"]}')
            plt.xlabel('Épocas')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
    
    plt.suptitle(f'Evolución del Accuracy durante el Entrenamiento - {idioma.upper()}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'resultados/graficas/evolucion_accuracy_{idioma}.png', dpi=300, bbox_inches='tight')
    plt.close()