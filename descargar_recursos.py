# descargar_recursos.py

import nltk

def descargar_recursos_nltk():
    """Descarga todos los recursos necesarios de NLTK"""
    recursos = ['stopwords', 'punkt']
    
    for recurso in recursos:
        try:
            nltk.download(recurso, quiet=False)
            print(f"✓ {recurso} descargado correctamente")
        except Exception as e:
            print(f"✗ Error descargando {recurso}: {e}")

if __name__ == '__main__':
    print("Descargando recursos de NLTK...")
    descargar_recursos_nltk()
    print("Listo!")