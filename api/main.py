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
films_fr = config.import_data("data/films_fr.csv")
films = config.import_data("data/films.csv")
musiques_import = config.import_data("data/musiques.csv")
livres_import_en = config.import_data("data/Livres_en_anglais.csv")
livres_toulouse_import = config.import_data("data/Librairie_toulouse.csv")
livres_import_fr = config.import_data("data/livres_fr.csv")

# Nétoyage des données
df_films_fr_not_na = config.traitement_na(films_fr)
df_films_not_na = config.traitement_na(films)
df_livres_fr_not_na = config.traitement_na(livres_import_fr)
livres_toulouse_not_na = config.traitement_na(livres_toulouse_import)
livres_en_not_na = config.traitement_na(livres_import_en)

df_musiques_not_na = config.traitement_na(musiques_import)

livres_toulouse_not_na.rename(columns={
    "year": "annee",
    "title": "titre",
    "author": "auteur",
    "publisher": "genre",
    "classification": "description",
    "library": "source"
}, inplace=True)

livres_en_not_na.rename(columns={
    "Year_published": "annee",
    "Original_Book_Title": "titre",
    "Author_Name": "auteur",
    "Genres": "genre",
    "Book_Description": "description",
    "Edition_Language": "langue"
}, inplace=True)

livres_en_not_na["source"] = livres_en_not_na.apply(lambda x: f"{x['annee']} - {x['titre']} - {x['auteur']}", axis=1)

df_films = pd.concat([df_films_fr_not_na, df_films_not_na], ignore_index=True)
df_livres = pd.concat([df_livres_fr_not_na, livres_toulouse_not_na, livres_en_not_na], ignore_index=True)
df_musiques = df_musiques_not_na

print("Colonnes du DataFrame :", list(df_films.isnull().sum()))


@app.get("/", tags=["Recommandations"])
async def home():
    return {"message": "Bienvenue sur l'API Chatbot Culture & Loisirs."}

#Rechercher un film
@app.get("/films/", tags=["Recommandations"])
async def get_films(titre: str = Query(None, description="Nom ou mot-clé du film")):
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
async def get_musiques(titre: str = Query(None, description="Nom ou mot-clé de la musique")):
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
async def get_livres1(titre: str = Query(None, description="Titre ou mot-clé du livre")):
    try:
        if titre:
            result = df_livres[df_livres["titre"].str.contains(titre, case=False, na=False)].head(10)
        else:
            result = df_livres.sample(10)
        return result.to_dict(orient="records")
    except Exception as e:
        print("Erreur : ", e)
