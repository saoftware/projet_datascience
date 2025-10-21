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
livres_import = config.import_data("data/Librairie_toulouse.csv")
livres_toulouse_import = config.import_data("data/Librairie_toulouse.csv")
livres_en_import = config.import_data("data/livres.csv")

# Nétoyage des données
df_films = config.traitement_na(films)
df_musiques = config.traitement_na(musiques_import)
df_livres_first = config.traitement_na(livres_import)
livres_toulouse = config.traitement_na(livres_import)
livres_en = config.traitement_na(livres_import)

livres_toulouse.rename(columns={
    "year": "annee",
    "title": "titre",
    "author": "auteur",
    "publisher": "genre",
    "classification": "description",
    "library": "source"
}, inplace=True)

livres_toulouse["langue"] = "français"

df_livres = pd.concat([df_livres_first, livres_toulouse], axis=1)


@app.get("/", tags=["Recommandations"])
async def home():
    return {"message": "Bienvenue sur l'API Chatbot Culture & Loisirs."}

#Rechercher un film
@app.get("/films/", tags=["Recommandations"])
async def get_films(titre: str = Query(None, description="Nom ou mot-clé du film")):
    if titre:
        result = df_films[df_films["titre"].str.contains(titre, case=False, na=False)].head(10)
    else:
        result = df_films.sample(10)
    return result.to_dict(orient="records")

# Rechercher une musique
@app.get("/musiques/", tags=["Recommandations"])
async def get_musiques(titre: str = Query(None, description="Nom ou mot-clé de la musique")):
    if titre:
        result = df_musiques[df_musiques["titre"].str.contains(titre, case=False, na=False)].head(10)
    else:
        result = df_musiques.sample(10)
    return result.to_dict(orient="records")

# Rechercher un livre
@app.get("/livres/", tags=["Recommandations"])
async def get_livres(titre: str = Query(None, description="Titre ou mot-clé du livre")):
    if titre:
        result = df_livres[df_livres["titre"].str.contains(titre, case=False, na=False)].head(10)
    else:
        result = df_livres.sample(10)
    return result.to_dict(orient="records")
