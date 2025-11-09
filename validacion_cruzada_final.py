def ejecutar_validacion_cruzada_completa():
    """Ejecuta validación cruzada para las 3 mejores configuraciones de cada idioma"""
    
    mejores_es = [9, 2, 8]  # Configs con mejor F1 en español
    mejores_en = [9, 2, 6]  # Configs con mejor F1 en inglés
    
    print("VALIDACIÓN CRUZADA 5-FOLDS")
    print("==========================")
    
    for idioma, mejores in [('es', mejores_es), ('en', mejores_en)]:
        print(f"\nIDIOMA: {idioma.upper()}")
        for config_idx in mejores:
            config = CONFIGURACIONES[config_idx-1]
            print(f"Config {config_idx}: {config}")
            # Ejecutar validación cruzada aquí