import requests
import pandas as pd
import os
import time


# Crée le dossier
os.makedirs("data", exist_ok=True)

API_KEY = "API_KEY"
BASE_URL = "https://api.themoviedb.org/3/discover/movie"


# Scraping des films
def collect_films(langue, pages):
    films = []
    for page in range(1, pages + 1):
        try:
            params = {
                "api_key": API_KEY,
                "language": langue,
                "sort_by": "popularity.desc",
                "page": page
            }
            response = requests.get(BASE_URL, params=params, timeout=10)
            data = response.json()

            if "results" not in data:
                print(f"Aucune donnée à la page {page}, arrêt.")
                break

            for f in data.get("results", []):
                info = f.get("volumeInfo", {})
                films.append({
                    "titre": f.get("title"),
                    "auteur": ", ".join(info.get("authors", [])) if "authors" in info else "Inconnu",
                    "langue": "français" if langue == "fr-FR" else "anglais",
                    "genre": ", ".join([str(g) for g in f.get("genre_ids", [])]),
                    "description": f.get("overview"),
                    "annee": f.get("release_date", "")[:4],
                    "source": f"https://www.themoviedb.org/movie/{f.get('id')}"
                })
            time.sleep(0.25)
        except Exception as e:
            print(f"Erreur page {page} : {e}")
            continue
    return pd.DataFrame(films)


# Scraping des livres
def collect_livres(langue, nb):
    livres = []
    base_url = "https://www.googleapis.com/books/v1/volumes"
    query = "livre" if langue == "français" else "book"
    max_results = 40  # limite par requête
    pages = nb // max_results

    for i in range(pages):
        params = {"q": query, "langRestrict": "fr" if langue == "français" else "en", "startIndex": i * max_results, "maxResults": max_results}
        r = requests.get(base_url, params=params)
        data = r.json()
        for item in data.get("items", []):
            info = item.get("volumeInfo", {})
            livres.append({
                "titre": info.get("title"),
                "auteur": ", ".join(info.get("authors", [])) if "authors" in info else "Inconnu",
                "langue": langue,
                "genre": ", ".join(info.get("categories", [])) if "categories" in info else "Non spécifié",
                "description": info.get("description", "Description non disponible"),
                "annee": info.get("publishedDate", "")[:4],
                "source": info.get("infoLink")
            })
    return pd.DataFrame(livres)


# Scraping des musiques
def collect_musiques(langue, nb):
    musiques = []
    base_url = "https://itunes.apple.com/search"
    pays = "fr" if langue == "français" else "us"
    term = "music"
    limit = 200  # max par requête
    pages = nb // limit

    for i in range(pages):
        params = {"term": term, "media": "music", "country": pays, "limit": limit, "offset": i*limit}
        r = requests.get(base_url, params=params)
        data = r.json()
        for item in data.get("results", []):
            musiques.append({
                "titre": item.get("trackName"),
                "artiste": item.get("artistName"),
                "album": item.get("collectionName"),
                "langue": langue,
                "genre": item.get("primaryGenreName"),
                "annee": item.get("releaseDate", "")[:4],
                "source": item.get("trackViewUrl")
            })
    return pd.DataFrame(musiques)


# Collecte musique
df_music_fr = collect_musiques("français", 2000)
df_music_en = collect_musiques("anglais", 1000)
df_music = pd.concat([df_music_fr, df_music_en], ignore_index=True)

# Sauvegarde
df_music.to_csv("data/musiques.csv", index=False, encoding="utf-8")
print(f"{len(df_music)} musiques enregistrées dans data/musiques.csv")

# Collecte
df_books_fr = collect_livres("français", 1000)
df_books_en = collect_livres("anglais", 1000)
df_books = pd.concat([df_books_fr, df_books_en], ignore_index=True)

# Sauvegarde
df_books.to_csv("data/livres.csv", index=False, encoding="utf-8")
print(f"{len(df_books)} livres enregistrés dans data/livres.csv")


# Récupérer 100 films français et 100 anglais
df_fr = collect_films("fr-FR", pages=100)
df_en = collect_films("en-US", pages=100)

# Fusionner et sauvegarder
df_films = pd.concat([df_en], ignore_index=True)
df_films.to_csv("data/films.csv", index=False, encoding="utf-8")

print(f"{len(df_films)} films sauvegardés dans data/films.csv")
