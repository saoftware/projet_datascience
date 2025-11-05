from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import sys

# Ajout du chemin pour accéder aux modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import data_cleaning, config

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


# Définir les chemins des fichiers de données nettoyées
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
films_path = os.path.join(BASE_DIR, "data/cleaned/films.csv")
livres_path = os.path.join(BASE_DIR, "data/cleaned/livres.csv")
musiques_path = os.path.join(BASE_DIR, "data/cleaned/musiques.csv")


# Vérifier si les fichiers existent, sinon les générer
if not (os.path.exists(films_path) and os.path.exists(livres_path) and os.path.exists(musiques_path)):
    print("Fichiers manquants : nettoyage et génération en cours...")
    data_cleaning.load_clean_and_save_data()
else:
    print("Fichiers déjà présents, chargement direct.")

# Charger les données
films_df = config.import_data(films_path)
livres_df = config.import_data(livres_path)
musiques_df = config.import_data(musiques_path)


@app.get("/", tags=["Recommandations"])
async def home():
    return {"message": "Bienvenue sur l'API Chatbot Culture & Loisirs."}


# Rechercher un film
@app.get("/films/", tags=["Recommandations"])
async def get_films(titre: str = Query(None, description="Titre du film à rechercher")):
    if titre:
        # Filtrer par titre
        results = films_df[films_df['titre'].str.contains(titre, case=False, na=False)]
        return results.to_dict(orient='records')
    else:
        # Retourner un échantillon aléatoire
        return films_df.sample(min(5, len(films_df))).to_dict(orient='records')

# Rechercher un livre
@app.get("/livres/", tags=["Recommandations"])
async def get_livres(titre: str = Query(None, description="Titre du livre à rechercher")):
    if titre:
        # Filtrer par titre
        results = livres_df[livres_df['titre'].str.contains(titre, case=False, na=False)]
        return results.to_dict(orient='records')
    else:
        # Retourner un échantillon aléatoire
        return livres_df.sample(min(5, len(livres_df))).to_dict(orient='records')

# Rechercher une musique
@app.get("/musiques/", tags=["Recommandations"])
async def get_musiques(titre: str = Query(None, description="Titre de la musique à rechercher")):
    if titre:
        # Filtrer par titre
        results = musiques_df[musiques_df['titre'].str.contains(titre, case=False, na=False)]
        return results.to_dict(orient='records')
    else:
        # Retourner un échantillon aléatoire
        return musiques_df.sample(min(5, len(musiques_df))).to_dict(orient='records')
