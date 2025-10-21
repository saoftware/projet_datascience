from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from modules import config

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

# Charger les datasets

films = config.import_data("data/films_fr.csv")
musiques_import = config.import_data("data/musiques.csv")
livres_import = config.import_data("data/livres.csv")

# Nétoyage des données
films = config.traitement_na(films)
musiques = config.traitement_na(musiques_import)
livres = config.traitement_na(livres_import)

@app.get("/", tags=["Recommandations"])
async def home():
    return {"message": "Bienvenue sur l'API Chatbot Culture & Loisirs."}

#Rechercher un film
@app.get("/films/", tags=["Recommandations"])
async def get_films(titre: str = Query(None, description="Nom ou mot-clé du film")):
    if titre:
        result = films[films["titre"].str.contains(titre, case=False, na=False)].head(10)
    else:
        result = films.sample(10)
    return result.to_dict(orient="records")

# Rechercher une musique
@app.get("/musiques/", tags=["Recommandations"])
async def get_musiques(titre: str = Query(None, description="Nom ou mot-clé de la musique")):
    if titre:
        result = musiques[musiques["titre"].str.contains(titre, case=False, na=False)].head(10)
    else:
        result = musiques.sample(10)
    return result.to_dict(orient="records")

# Rechercher un livre
@app.get("/livres/", tags=["Recommandations"])
async def get_livres(titre: str = Query(None, description="Titre ou mot-clé du livre")):
    if titre:
        result = livres[livres["titre"].str.contains(titre, case=False, na=False)].head(10)
    else:
        result = livres.sample(10)
    return result.to_dict(orient="records")
