import pandas as pd
import modules.data_cleaning as data_cleaning
import modules.config as config
import os


# Dossiers / fichiers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "data_cleaned")
FILMS_PATH = os.path.join(DATA_DIR, "films.csv")
LIVRES_PATH = os.path.join(DATA_DIR, "livres.csv")
MUSIQUES_PATH = os.path.join(DATA_DIR, "musiques.csv")

# Vérifier si les fichiers existent
if not (os.path.exists(FILMS_PATH) and os.path.exists(LIVRES_PATH) and os.path.exists(MUSIQUES_PATH)):
    print("Fichiers manquants : nettoyage et génération en cours...")
    data_cleaning.load_clean_and_save_data()
else:
    print("Fichiers déjà présents, chargement direct.")

# Utilisation de chemins absolus pour le chargement des données
df_films = config.import_data(FILMS_PATH)
df_livres = config.import_data(LIVRES_PATH)
df_musiques = config.import_data(MUSIQUES_PATH)


#Rechercher un film
def films_recommandations(titre: str):
    try:
        if titre:
            result = df_films[df_films["titre"].str.contains(titre, case=False, na=False)].head(10)
        else:
            result = df_films.sample(10)
            
        # Supprimer les doublons
        result = result.drop_duplicates(subset="titre")
        
        return result.to_dict(orient="records")
    except Exception as e:
        print("Erreur : ", e)


# Rechercher une musique
def musiques_recommandations(titre: str):
    try:
        if titre:
            result = df_musiques[df_musiques["titre"].str.contains(titre, case=False, na=False)].head(10)
        else:
            result = df_musiques.sample(10)
            
        # Supprimer les doublons
        result = result.drop_duplicates(subset="titre")
        
        return result.to_dict(orient="records")
    except Exception as e:
        print("Erreur : ", e)


# Rechercher un livre
def livres_recommandations(titre: str):
    try:
        if titre:
            result = df_livres[df_livres["titre"].str.contains(titre, case=False, na=False)].head(10)
        else:
            result = df_livres.sample(10)
        
        return result.to_dict(orient="records")
    except Exception as e:
        print("Erreur : ", e)
