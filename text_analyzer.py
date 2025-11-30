import os
import re
import spacy
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
nlp = spacy.load('es_core_news_sm')
#Carga el txt
def cargar_texto():
    archivo = "lotr.txt"
    with open(archivo, "r", encoding="utf-8") as file:
        return file.read()

texto = cargar_texto()
# Funciones para la normalización
def remove_punctuation(words):
    #Elimina los signos de puntuación
    doc = spacy.tokens.doc.Doc(nlp.vocab, words=words)
    return [word.text for word in doc if not word.is_punct]

def remove_stop_words(words):
    #Elimina las stopwords
    doc = nlp(" ".join(words))
    return [word.text for word in doc if not word.is_stop]

def remove_urls_and_emojis(text):
    #Elimina URLs y emojis de un texto.
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Eliminar URLs
    text = re.sub(r'[\U00010000-\U0010FFFF]', '', text, flags=re.UNICODE)  # Eliminar emojis
    return text

def to_lowercase(words):
    #Pasa el texto a minúsculas
    return [word.lower() for word in words]

def lemmatizer(words):
    #Devuelve el lema de las palabras
    doc = nlp(" ".join(words))
    return [word.lemma_ for word in doc]

def normalize(text):
    #Devuelve el texto tokenizado y normalizado
    text = remove_urls_and_emojis(text)
    # Tokeniza el texto con SpaCy
    doc = nlp(text)
    words = [token.text for token in doc]
    
    # Procesa las palabras
    words = remove_punctuation(words)
    words = remove_stop_words(words)
    words = to_lowercase(words)
    words = lemmatizer(words)
    return words
file_path = "lotr.txt"
if os.path.exists(file_path):
    raw_text = cargar_texto()
    normalized_text = normalize(raw_text)

    # Guarda el texto preprocesado
    with open("lotr_procesado.txt", "w", encoding="utf-8") as file:
        file.write(" ".join(normalized_text))

# Tokeniza y calcula la frecuencia de las palabras
def calcular_frecuencias(texto):
    tokens = texto.split()
    return Counter(tokens)

frecuencias = calcular_frecuencias(" ".join(normalized_text))

# Obtiene las 15 palabras más comunes
palabras_comunes = frecuencias.most_common(15)

# Saca un gráfico de barras
def graficar_frecuencias(palabras_comunes):
    palabras, cantidades = zip(*palabras_comunes)
    plt.bar(palabras, cantidades)
    plt.title("Palabras más comunes")
    plt.xlabel("Palabras")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=45)
    plt.show()

graficar_frecuencias(palabras_comunes)


# Identifica entidades nombradas y las categoriza
def extraer_entidades(doc, etiquetas_interes):
    entidades = []
    for ent in doc.ents:
        if ent.label_ in etiquetas_interes:
            entidades.append({"Texto": ent.text, "Etiqueta": ent.label_})
    return entidades

doc = nlp(texto)
entidades = extraer_entidades(doc, ["PER", "LOC", "ORG"])
entidades_df = pd.DataFrame(entidades)

#Hace que cada categoría tenga un índice personalizado
def personalizar_indices(df):
    contador = {"PER": 0, "LOC": 0, "ORG": 0}
    indices_personalizados = []

    for _, row in df.iterrows():
        etiqueta = row["Etiqueta"]
        if etiqueta in contador:
            contador[etiqueta] += 1
            indices_personalizados.append(f"{etiqueta}-{contador[etiqueta]}")
        else:
            indices_personalizados.append("OTRO")  # Por si aparecen etiquetas no contempladas

    df["Índice"] = indices_personalizados
    df.set_index("Índice", inplace=True)
    return df

entidades_df = personalizar_indices(entidades_df)


# Guarda las entidades en un archivo .csv
def guardar_csv(df, nombre_archivo):
    df.to_csv(nombre_archivo, encoding="utf-8")

guardar_csv(entidades_df, "entidades.csv")

# Calcula la proporción de las entidades 
def graficar_proporcion_entidades(df):
    proporcion_entidades = df["Etiqueta"].value_counts()
    proporcion_entidades.plot.pie(autopct="%1.1f%%", title="Proporción de Entidades Nombradas")
    plt.ylabel("")
    plt.show()

graficar_proporcion_entidades(entidades_df)
def guardar_csv(df, nombre_archivo):
    df.to_csv(nombre_archivo, encoding="utf-8")

guardar_csv(entidades_df, "entidades.csv")

def calcular_estadisticas(doc):
    return Counter(token.pos_ for token in doc)

estadisticas = calcular_estadisticas(doc)

# Crea una tabla con las estadísticas
def crear_tabla_estadisticas(estadisticas):
    return pd.DataFrame.from_dict(estadisticas, orient="index", columns=["Frecuencia"])

estadisticas_df = crear_tabla_estadisticas(estadisticas)

# Lo muestra con un gráfico de barras
def graficar_estadisticas(df):
    df.sort_values("Frecuencia", ascending=False).plot.bar(title="Frecuencia por tipo gramatical")
    plt.xlabel("Categoría gramatical")
    plt.ylabel("Frecuencia")
    plt.show()

graficar_estadisticas(estadisticas_df)

# Guarda las estadísticas en un archivo
guardar_csv(estadisticas_df, "estadisticas_gramaticales.csv")
