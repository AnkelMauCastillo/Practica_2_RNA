# src/preprocesamiento.py 

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import os

class Preprocesador:
    def __init__(self, idioma='es'):
        self.idioma = idioma
        self.stopwords = self._cargar_stopwords(idioma)
        self.stemmer = SnowballStemmer(idioma) if idioma in ['spanish', 'english'] else None

    def _cargar_stopwords(self, idioma):
        """Carga stopwords con manejo de errores"""
        try:
            # Verificar si las stopwords están disponibles
            if idioma == 'es':
                lang = 'spanish'
            elif idioma == 'en':
                lang = 'english'
            else:
                lang = idioma
                
            return set(stopwords.words(lang))
        except (LookupError, OSError):
            print(f"Stopwords para {idioma} no encontradas. Descargando...")
            try:
                nltk.download('stopwords', quiet=False)
                return set(stopwords.words(lang))
            except:
                print(f"No se pudieron cargar stopwords para {idioma}. Usando lista vacía.")
                return set()

    def limpiar_texto(self, texto):
        """Limpia el texto de URLs, menciones y puntuación"""
        if not isinstance(texto, str):
            return ""
            
        # Minúsculas
        texto = texto.lower()
        # Eliminar URLs
        texto = re.sub(r'http\S+', '', texto)
        # Eliminar menciones de usuario
        texto = re.sub(r'@\w+', '', texto)
        # Eliminar caracteres especiales pero mantener letras, números y espacios
        texto = re.sub(r'[^\w\s]', ' ', texto)
        # Eliminar números
        texto = re.sub(r'\d+', '', texto)
        # Eliminar espacios múltiples
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    def remover_stopwords(self, tokens):
        """Remueve stopwords de una lista de tokens"""
        return [t for t in tokens if t and t not in self.stopwords]

    def aplicar_stemming(self, tokens):
        """Aplica stemming a una lista de tokens"""
        if self.stemmer:
            return [self.stemmer.stem(t) for t in tokens if t]
        return tokens

    def preprocesar(self, texto, usar_stopwords=True, usar_stemming=True):
        """Aplica todo el pipeline de preprocesamiento"""
        texto_limpio = self.limpiar_texto(texto)
        tokens = texto_limpio.split()
        
        if usar_stopwords and self.stopwords:
            tokens = self.remover_stopwords(tokens)
            
        if usar_stemming and self.stemmer:
            tokens = self.aplicar_stemming(tokens)
            
        return ' '.join(tokens)