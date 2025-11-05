import pandas as pd
import modules.config as config

# Chargement, netoyage et sauvegarde des données
def load_clean_and_save_data():
    """
    Charge, nettoie et sauvegarde les données dans le dossier data_cleaned
    """
    import os
    # Chemin absolu vers le dossier du projet
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Charger les datasets avec chemins absolus
    films_fr_import = config.import_data(os.path.join(BASE_DIR, "data/films_fr.csv"))
    films_import = config.import_data(os.path.join(BASE_DIR, "data/films.csv"))
    musiques_import = config.import_data(os.path.join(BASE_DIR, "data/musiques.csv"))
    livres_en_import = config.import_data(os.path.join(BASE_DIR, "data/Livres_en_anglais.csv"))
    livres_toulouse_import = config.import_data(os.path.join(BASE_DIR, "data/Librairie_toulouse.csv"))
    livres_fr_import = config.import_data(os.path.join(BASE_DIR, "data/livres_fr.csv"))

    livres_toulouse_import.rename(columns={
        "year": "annee",
        "title": "titre",
        "author": "auteur",
        "classification": "genre",
        "publisher": "description",
        "library": "source"
    }, inplace=True)

    livres_toulouse_import["langue"] = "français"

    livres_en_import.rename(columns={
        "Year_published": "annee",
        "Original_Book_Title": "titre",
        "Author_Name": "auteur",
        "Genres": "genre",
        "Book_Description": "description",
        "Edition_Language": "langue"
    }, inplace=True)

    livres_en_import["source"] = livres_en_import.apply(lambda x: f"{x['annee']} - {x['titre']} - {x['auteur']}", axis=1)

    df_films_save = pd.concat([films_fr_import, films_import], ignore_index=True)
    df_livres_save = pd.concat([livres_fr_import, livres_toulouse_import, livres_en_import], ignore_index=True)
    df_musiques_save = musiques_import


    # Nétoyage des données
    df_films_save = config.traitement_na(df_films_save)
    df_livres_save = config.traitement_na(df_livres_save)
    df_musiques_save = config.traitement_na(df_musiques_save)
    
    df_films = config.drop_doublon(df_films_save)
    df_livres = config.drop_doublon(df_livres_save)
    df_musiques = config.drop_doublon(df_musiques_save)
    
    # Création du dossier cleaned s'il n'existe pas
    cleaned_dir = os.path.join(BASE_DIR, "data/cleaned")
    if not os.path.exists(cleaned_dir):
        os.makedirs(cleaned_dir)
        
    # Sauvegarde des données nettoyées
    df_films_save.to_csv(os.path.join(cleaned_dir, "films.csv"), index=False, encoding="utf-8")
    df_livres.to_csv(os.path.join(cleaned_dir, "livres.csv"), index=False, encoding="utf-8")
    df_musiques.to_csv(os.path.join(cleaned_dir, "musiques.csv"), index=False, encoding="utf-8")

    print(f"Films sauvegardés avec succès : {df_films.shape[0]} lignes")
    print(f"Livres sauvegardés avec succès : {df_livres.shape[0]} lignes")
    print(f"Musiques sauvegardées avec succès : {df_musiques.shape[0]} lignes")

