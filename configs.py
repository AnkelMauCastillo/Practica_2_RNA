# configs_completas.py

CONFIGURACIONES= [
    # Variar neuronas ocultas
    {'neuronas_ocultas': 64, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    {'neuronas_ocultas': 512, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    {'neuronas_ocultas': 1024, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    
    # Variar inicialización
    {'neuronas_ocultas': 128, 'inicializacion': 'normal', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    
    # Variar pesado de términos
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tfidf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    
    # Variar n-gramas
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (2,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    
    # Variar preprocesamiento
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar_sin_stopwords', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar_sin_stopwords_stemming', 'lr': 0.01, 'batch_size': 16, 'epochs': 100},
    
    # Variar learning rate
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.1, 'batch_size': 16, 'epochs': 100},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.5, 'batch_size': 16, 'epochs': 100},
    
    # Variar batch size
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 64, 'epochs': 100},
]