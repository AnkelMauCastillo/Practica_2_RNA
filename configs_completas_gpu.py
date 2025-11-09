CONFIGURACIONES = [
    # Variar neuronas ocultas (optimizadas para GPU)
    {'neuronas_ocultas': 64, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 200},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 200},
    {'neuronas_ocultas': 256, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 200},
    {'neuronas_ocultas': 512, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 64, 'epochs': 200},
    {'neuronas_ocultas': 1024, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 150},
    
    # Variar inicialización
    {'neuronas_ocultas': 128, 'inicializacion': 'normal', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 200},
    
    # Variar pesado de términos
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tfidf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 200},
    
    # Variar n-gramas
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (2,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 200},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 200},
    
    # Variar preprocesamiento
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar_sin_stopwords', 'lr': 0.01, 'batch_size': 128, 'epochs': 200},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar_sin_stopwords_stemming', 'lr': 0.01, 'batch_size': 128, 'epochs': 200},
    
    # Variar learning rate
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.1, 'batch_size': 128, 'epochs': 150},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.5, 'batch_size': 128, 'epochs': 100},
    
    # Variar batch size
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 200},
    {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
     'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 64, 'epochs': 200},
]