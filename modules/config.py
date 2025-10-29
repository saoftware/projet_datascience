import pandas as pd
import os, csv
from io import StringIO, BytesIO

# Fonction d'import des données
def import_data(path):
    """
    Charge un fichier CSV ou Excel (chemin local ou fichier uploadé)
    et renvoie un DataFrame propre.
    """
    df = None
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']

    if isinstance(path, str):
        filename = os.path.basename(path)
        file_extension = os.path.splitext(filename)[1].lower()
        file_source = path  # lecture directe du disque
    else:
        filename = path.name
        file_extension = os.path.splitext(filename)[1].lower()
        file_bytes = path.read()
        file_source = BytesIO(file_bytes)

    # Traitement selon l'extension
    if file_extension == ".csv":
        for enc in encodings:
            try:
                # On lit quelques lignes pour détecter le séparateur
                if isinstance(file_source, str):
                    with open(file_source, 'r', encoding=enc, errors='ignore') as f:
                        sample = f.readline()
                else:
                    sample = file_source.getvalue().decode(enc, errors='ignore').splitlines()[0]

                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                sep = dialect.delimiter

                df = pd.read_csv(file_source, encoding=enc, sep=sep)
                break
            except Exception as e:
                continue

    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(file_source, sheet_name=0)

    else:
        raise ValueError(f"Format non pris en charge : {file_extension}")
    
    if df is None:
        raise ValueError(f"Impossible de charger le fichier {filename}")

    print(f"Fichier chargé : {filename} ({len(df)} lignes, {len(df.columns)} colonnes)")
    return df


# Traitement des valeurs manquantes
def traitement_na(data):
    try:
        num_cols = data.select_dtypes(exclude="object")
        cat_cols = data.select_dtypes(include="object")
        df = pd.concat([cat_cols, num_cols], axis=1)
        percent_na = (df.isnull().sum() / len(df)) * 100
        
        for col in df.columns:
            if percent_na[col] > 20:
                df.drop(columns=[col], inplace=True)
            elif percent_na[col] > 0:
                if df[col].dtype == "object":
                    df.fillna({col: "Inconnu"}, inplace=True)
                else:
                    df.fillna({col: df[col].median()}, inplace=True)
                
        return df
    
    except Exception as e:
        print("Erreur de chargement :", e)
        
        
# suppression des données dupliquées
def drop_doublon(data):
    try:
        # Supprimer les doublons
        result = data.drop_duplicates(subset="titre")
        return result
    
    except Exception as e:
        print("Erreur de chargement :", e)