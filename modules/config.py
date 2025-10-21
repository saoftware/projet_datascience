import pandas as pd
import os, csv


# Fonction d'import des fichiers
def import_data(path: str):
    """Charge un Fichier CSVExcel et renvoie un DataFrame."""
    df = None  # Initialisation

    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print("Erreur de chargement : {e}")
    


# Traitement des valeurs manquantes
def traitement_na(data):
    try:
        num_cols = data.select_dtypes(exclude="object")
        cat_cols = data.select_dtypes(include="object")
        df = pd.concat([cat_cols, num_cols], axis=1)
        percent_na = (df.isnull().sum() / len(df)) * 100
        
        for col in df.columns:
            if percent_na[col] > 0.2:
                df.drop(columns=[col], inplace=True)
            elif percent_na[col] > 0:
                if df[col].dtype == "object":
                    df[col].fillna("Inconnu", inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
                    

        return df
    
    except Exception as e:
        print("Erreur de chargement :", e)
        
        
# Fusion des données
def fusion_data(data):
    try:
        num_cols = data.select_dtypes(exclude="object")
        cat_cols = data.select_dtypes(include="object")
        df_fusion = pd.concat([cat_cols, num_cols], axis=1)
            
        return df_fusion
    
    except Exception as e:
        print("Erreur de chargement :", e)