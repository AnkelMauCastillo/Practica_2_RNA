CONFIGURACIONES_CV = {
    'es': [
        # Las 3 mejores configuraciones para español
        {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
         'ngramas': (1,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 100},
        {'neuronas_ocultas': 64, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
         'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 100},
        {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
         'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 100}
    ],
    'en': [
        # Las 3 mejores configuraciones para inglés
        {'neuronas_ocultas': 128, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
         'ngramas': (1,2), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 100},
        {'neuronas_ocultas': 64, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
         'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 128, 'epochs': 100},
        {'neuronas_ocultas': 1024, 'inicializacion': 'xavier', 'pesado_terminos': 'tf', 
         'ngramas': (1,1), 'preprocesamiento': 'normalizar', 'lr': 0.01, 'batch_size': 32, 'epochs': 100}
    ]
}