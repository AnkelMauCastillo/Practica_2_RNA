# src/representaciones.py

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def crear_vectorizador(tipo='tf', ngram_range=(1,1)):
    if tipo == 'tf':
        return CountVectorizer(ngram_range=ngram_range)
    elif tipo == 'tfidf':
        return TfidfVectorizer(ngram_range=ngram_range)
    else:
        raise ValueError("Tipo debe ser 'tf' o 'tfidf'")