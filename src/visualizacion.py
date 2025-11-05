# src/visualizacion.py

import matplotlib.pyplot as plt
import numpy as np

def graficar_perdidas(configuraciones, resultados, top_n=5):
    """Grafica las curvas de pérdida para las mejores configuraciones"""
    
    # Ordenar por F1-score y tomar las mejores
    indices_mejores = np.argsort([r['f1'] for r in resultados])[-top_n:][::-1]
    
    plt.figure(figsize=(12, 8))
    
    for i, idx in enumerate(indices_mejores):
        config = configuraciones[idx]
        resultado = resultados[idx]
        
        plt.subplot(2, 3, i+1)
        plt.plot(resultado['train_losses'], label='Train Loss', alpha=0.7)
        plt.plot(resultado['test_losses'], label='Test Loss', alpha=0.7)
        plt.title(f"Config {idx+1}\nF1: {resultado['f1']:.3f}")
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados/graficas/mejores_configuraciones.png', dpi=300, bbox_inches='tight')
    plt.show()

def graficar_metricas_comparativas(resultados, configuraciones):
    """Grafica comparativa de métricas para todas las configuraciones"""
    f1_scores = [r['f1'] for r in resultados]
    precision_scores = [r['precision'] for r in resultados]
    recall_scores = [r['recall'] for r in resultados]
    
    x = np.arange(len(resultados))
    width = 0.25
    
    plt.figure(figsize=(15, 6))
    
    plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.7)
    plt.bar(x, recall_scores, width, label='Recall', alpha=0.7)
    plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.7)
    
    plt.xlabel('Configuraciones')
    plt.ylabel('Score')
    plt.title('Comparación de Métricas por Configuración')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x, [f'Config {i+1}' for i in range(len(resultados))], rotation=45)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('resultados/graficas/comparacion_metricas.png', dpi=300, bbox_inches='tight')
    plt.show()

def generar_tabla_resultados(resultados, configuraciones):
    """Genera una tabla con los resultados"""
    print("\n" + "="*100)
    print("TABLA COMPLETA DE RESULTADOS")
    print("="*100)
    print(f"{'Config':^8} {'Neuronas':^10} {'Inicial':^10} {'Pesado':^8} {'Ngramas':^12} {'Preproc':^20} {'LR':^8} {'Batch':^8} {'F1':^8} {'Precision':^10} {'Recall':^8}")
    print("-"*100)
    
    for i, (config, res) in enumerate(zip(configuraciones, resultados)):
        print(f"{i+1:^8} {config['neuronas_ocultas']:^10} {config['inicializacion']:^10} "
              f"{config['pesado_terminos']:^8} {str(config['ngramas']):^12} {config['preprocesamiento']:^20} "
              f"{config['lr']:^8} {config['batch_size']:^8} {res['f1']:^8.3f} {res['precision']:^10.3f} {res['recall']:^8.3f}")
    
    # Guardar tabla en archivo
    with open('resultados/tabla_resultados.txt', 'w', encoding='utf-8') as f:
        f.write("TABLA COMPLETA DE RESULTADOS\n")
        f.write("="*100 + "\n")
        f.write(f"{'Config':^8} {'Neuronas':^10} {'Inicial':^10} {'Pesado':^8} {'Ngramas':^12} {'Preproc':^20} {'LR':^8} {'Batch':^8} {'F1':^8} {'Precision':^10} {'Recall':^8}\n")
        f.write("-"*100 + "\n")
        
        for i, (config, res) in enumerate(zip(configuraciones, resultados)):
            f.write(f"{i+1:^8} {config['neuronas_ocultas']:^10} {config['inicializacion']:^10} "
                   f"{config['pesado_terminos']:^8} {str(config['ngramas']):^12} {config['preprocesamiento']:^20} "
                   f"{config['lr']:^8} {config['batch_size']:^8} {res['f1']:^8.3f} {res['precision']:^10.3f} {res['recall']:^8.3f}\n")