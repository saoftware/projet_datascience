from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from modules import data_cleaning, config
import os

app = FastAPI(
    title="Chatbot Culture & Loisirs API",
    version="1.0.0",
    description="APIs gestion de recommandations de films, musiques et livres."
)

# Autoriser la communication avec ton interface web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dossiers / fichiers
DATA_DIR = "data/data_cleaned"
FILMS_PATH = os.path.join(DATA_DIR, "films.csv")
LIVRES_PATH = os.path.join(DATA_DIR, "livres.csv")
MUSIQUES_PATH = os.path.join(DATA_DIR, "musiques.csv")


# Vérifier si les fichiers existent
if not (os.path.exists(FILMS_PATH) and os.path.exists(LIVRES_PATH) and os.path.exists(MUSIQUES_PATH)):
    print("Fichiers manquants : nettoyage et génération en cours...")
    data_cleaning.load_clean_and_save_data()
else:
    print("Fichiers déjà présents, chargement direct.")


# Chargement des données saugardées
df_films = config.import_data("data/data_cleaned/films.csv")
df_livres = config.import_data("data/data_cleaned/livres.csv")
df_musiques  = config.import_data("data/data_cleaned/musiques.csv")


@app.get("/", tags=["Recommandations"])
async def home():
    return {"message": "Bienvenue sur l'API Chatbot Culture & Loisirs."}


#Rechercher un film
@app.get("/films/", tags=["Recommandations"])
async def films_recommandations(titre: str = Query(None, description="Nom ou mot-clé du film")):
    try:
        if titre:
            result = df_films[df_films["titre"].str.contains(titre, case=False, na=False)].head(10)
        else:
            result = df_films.sample(10)
        
        return result.to_dict(orient="records")
    except Exception as e:
        print("Erreur : ", e)


# Rechercher une musique
@app.get("/musiques/", tags=["Recommandations"])
async def musiques_recommandations(titre: str = Query(None, description="Nom ou mot-clé de la musique")):
    try:
        if titre:
            result = df_musiques[df_musiques["titre"].str.contains(titre, case=False, na=False)].head(10)
        else:
            result = df_musiques.sample(10)
        
        return result.to_dict(orient="records")
    except Exception as e:
        print("Erreur : ", e)


# Rechercher un livre
@app.get("/livres/", tags=["Recommandations"])
async def livres_recommandations(titre: str = Query(None, description="Titre ou mot-clé du livre")):
    try:
        if titre:
            result = df_livres[df_livres["titre"].str.contains(titre, case=False, na=False)].head(10)
        else:
            result = df_livres.sample(10)
        
        return result.to_dict(orient="records")
    except Exception as e:
        print("Erreur : ", e)
