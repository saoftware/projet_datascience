import streamlit as st
import pandas as pd
import requests
import os
import sys

# Ajout du chemin du projet pour pouvoir importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'projet_datascience'))

# Import des modules du projet
try:
    from modules import recommandation, config, data_cleaning
    MODULES_LOADED = True
    print("Modules chargés avec succès!")
except ImportError as e:
    print(f"Erreur lors de l'importation des modules: {e}")
    MODULES_LOADED = False

# Définition de l'URL de l'API
API_URL = "http://127.0.0.1:8000"  # FastAPI utilise le port 8000 par défaut

# Fonction pour vérifier si l'API est disponible
def is_api_available():
    try:
        # Utiliser un timeout court pour éviter de bloquer trop longtemps
        response = requests.get(f"{API_URL}/films/", timeout=1)
        if response.status_code == 200:
            print("API connectée avec succès!")
            return True
        else:
            print(f"API a répondu avec le code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Erreur de connexion à l'API: {e}")
        return False

# Vérification de la disponibilité de l'API
API_AVAILABLE = is_api_available()
print(f"API disponible: {API_AVAILABLE}")

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Système de Recommandation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et description
st.title("Système de Recommandation Intelligent")
st.markdown("""
Ce dashboard vous permet d'interagir avec notre système de recommandation pour obtenir des suggestions 
personnalisées de livres, films ou musiques basées sur vos préférences.
""")

# Sidebar pour les options utilisateur
st.sidebar.header("Options")

# Choix du type de contenu
content_type = st.sidebar.selectbox(
    "Choisissez le type de contenu",
    ["Livres", "Films", "Musiques"]
)

# Fonction pour charger les données selon le type de contenu
@st.cache_data
def load_data(content_type):
    if MODULES_LOADED:
        # Utilisation directe des dataframes du module de recommandation
        if content_type == "Livres":
            return recommandation.df_livres
        elif content_type == "Films":
            return recommandation.df_films
        else:  # Musiques
            return recommandation.df_musiques
    else:
        # Fallback si les modules ne sont pas chargés
        if content_type == "Livres":
            try:
                df_fr = pd.read_csv("data/livres_fr.csv")
                df_en = pd.read_csv("data/Livres_en_anglais.csv")
                return pd.concat([df_fr, df_en], ignore_index=True)
            except:
                return pd.read_csv("data/livres_fr.csv")
        elif content_type == "Films":
            try:
                df_fr = pd.read_csv("data/films_fr.csv")
                return df_fr
            except:
                return pd.read_csv("data/films.csv")
        else:  # Musiques
            return pd.read_csv("data/musiques.csv")

# Chargement des données
try:
    df = load_data(content_type)
    st.sidebar.success(f"Données {content_type} chargées avec succès!")
except Exception as e:
    st.sidebar.error(f"Erreur lors du chargement des données: {e}")
    df = pd.DataFrame()

# Affichage d'un aperçu des données
if not df.empty:
    st.subheader(f"Aperçu des données ({content_type})")
    st.dataframe(df.head())

# Options de recherche
st.subheader("Recherche et Recommandation")

search_method = st.radio(
    "Méthode de recherche",
    ["Par mots-clés", "Par titre similaire", "Par recommandation personnalisée"]
)

if search_method == "Par mots-clés":
    keywords = st.text_input("Entrez des mots-clés séparés par des espaces")
    col1, col2 = st.columns([1,1])

    with col2:
        if st.button("Réinitialiser la discussion"):
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()

    with col1:
        if st.button("Rechercher") and keywords:
            st.info("Recherche en cours...")
            # Simulation de recherche par mots-clés
            # Dans une implémentation réelle, vous appelleriez votre API ici
            try:
                # Exemple de filtrage simple (à remplacer par appel API)
                results = df[df.apply(lambda row: any(kw.lower() in str(row).lower() for kw in keywords.split()), axis=1)]
                if not results.empty:
                    st.success(f"{len(results)} résultats trouvés")
                    st.dataframe(results)
                else:
                    st.warning("Aucun résultat trouvé pour ces mots-clés.")
            except Exception as e:
                st.error(f"Erreur lors de la recherche: {e}")

elif search_method == "Par titre similaire":
    if not df.empty:
        # Récupération des titres selon le type de contenu
        title_column = "titre" if "titre" in df.columns else "title" if "title" in df.columns else df.columns[0]
        titles = df[title_column].dropna().unique().tolist()
        selected_title = st.selectbox("Sélectionnez un titre", titles)
        
        if st.button("Trouver des titres similaires"):
            st.info("Recherche de titres similaires...")
            try:
                # Essayer d'utiliser l'API d'abord si elle est disponible
                if API_AVAILABLE:
                    endpoint = ""
                    if content_type == "Livres":
                        endpoint = f"{API_URL}/livres/?titre={selected_title}"
                    elif content_type == "Films":
                        endpoint = f"{API_URL}/films/?titre={selected_title}"
                    else:  # Musiques
                        endpoint = f"{API_URL}/musiques/?titre={selected_title}"
                    
                    response = requests.get(endpoint)
                    if response.status_code == 200:
                        results = response.json()
                        if results:
                            st.success(f"Titres similaires à '{selected_title}' (via API)")
                            for i, item in enumerate(results, 1):
                                st.write(f"{i}. {item['titre']}")
                        else:
                            st.warning("Aucun titre similaire trouvé via l'API.")
                    else:
                        st.error(f"Erreur lors de l'appel à l'API: {response.status_code}")
                        # Fallback aux modules si l'API échoue
                        if MODULES_LOADED:
                            st.info("Utilisation des modules locaux comme alternative...")
                
                # Utiliser les modules si l'API n'est pas disponible ou a échoué
                elif MODULES_LOADED:
                    # Utilisation des fonctions du module de recommandation
                    if content_type == "Livres":
                        results = recommandation.livres_recommandations(selected_title)
                        if results:
                            st.success(f"Titres similaires à '{selected_title}' (via modules)")
                            for i, item in enumerate(results, 1):
                                st.write(f"{i}. {item['titre']}")
                        else:
                            st.warning("Aucun titre similaire trouvé.")
                    elif content_type == "Films":
                        results = recommandation.films_recommandations(selected_title)
                        if results:
                            st.success(f"Titres similaires à '{selected_title}' (via modules)")
                            for i, item in enumerate(results, 1):
                                st.write(f"{i}. {item['titre']}")
                        else:
                            st.warning("Aucun titre similaire trouvé.")
                    else:  # Musiques
                        results = recommandation.musiques_recommandations(selected_title)
                        if results:
                            st.success(f"Titres similaires à '{selected_title}' (via modules)")
                            for i, item in enumerate(results, 1):
                                st.write(f"{i}. {item['titre']}")
                        else:
                            st.warning("Aucun titre similaire trouvé.")
                else:
                    # Fallback si ni l'API ni les modules ne sont disponibles
                    st.success(f"Titres similaires à '{selected_title}' (simulation)")
                    # Affichage de 5 titres aléatoires comme exemple
                    import random
                    similar_titles = random.sample(titles, min(5, len(titles)))
                    for i, title in enumerate(similar_titles, 1):
                        st.write(f"{i}. {title}")
            except Exception as e:
                st.error(f"Erreur lors de la recherche: {e}")

elif search_method == "Par recommandation personnalisée":
    st.write("Entrez vos préférences pour obtenir des recommandations personnalisées")
    
    # Champs de préférences selon le type de contenu
    if content_type == "Livres":
        genre = st.multiselect("Genres préférés", ["Roman", "Science-fiction", "Fantastique", "Policier", "Biographie", "Histoire"])
        auteur = st.text_input("Auteurs préférés (séparés par des virgules)")
    elif content_type == "Films":
        genre = st.multiselect("Genres préférés", ["Action", "Comédie", "Drame", "Science-fiction", "Horreur", "Documentaire"])
        realisateur = st.text_input("Réalisateurs préférés (séparés par des virgules)")
    else:  # Musiques
        genre = st.multiselect("Genres préférés", ["Pop", "Rock", "Hip-hop", "Jazz", "Classique", "Électronique"])
        artiste = st.text_input("Artistes préférés (séparés par des virgules)")
    
    if st.button("Obtenir des recommandations"):
        st.info("Génération de recommandations personnalisées...")
        # Simulation de recommandations personnalisées
        # Dans une implémentation réelle, vous appelleriez votre API ici
        try:
            # Exemple simple (à remplacer par appel API)
            st.success("Recommandations personnalisées")
            # Affichage de 5 titres aléatoires comme exemple
            if not df.empty:
                title_column = "titre" if "titre" in df.columns else "title" if "title" in df.columns else df.columns[0]
                recommendations = df.sample(min(5, len(df)))
                st.dataframe(recommendations[[title_column] + [col for col in recommendations.columns if col != title_column][:3]])
        except Exception as e:
            st.error(f"Erreur lors de la génération des recommandations: {e}")

# Section pour le chatbot
st.title("Assistant de Recommandation")

# Initialisation de l'historique de chat dans la session state si non existant
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre assistant de recommandation. Comment puis-je vous aider aujourd'hui ?"}
    ]

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Zone de saisie du message utilisateur
user_input = st.chat_input("Posez votre question ici...")

# Traitement du message utilisateur
if user_input:
    # Ajout du message utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Affichage du message utilisateur
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Traitement de la réponse
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Extraction de mots-clés du message utilisateur (simplifiée)
            keywords = user_input.lower().split()
            search_term = " ".join(keywords[:2]) if len(keywords) > 1 else keywords[0] if keywords else ""
            
            # Simulation d'une réponse progressive
            message_placeholder.markdown("⏳ Réflexion en cours...")
            
            # Préparation de la réponse
            if content_type == "Livres":
                intro = f"Voici des recommandations de livres basées sur votre demande concernant '{search_term}' :"
            elif content_type == "Films":
                intro = f"Voici des recommandations de films basées sur votre demande concernant '{search_term}' :"
            else:  # Musiques
                intro = f"Voici des recommandations de musiques basées sur votre demande concernant '{search_term}' :"
            
            # Essayer d'utiliser l'API d'abord si elle est disponible
            results = []
            if API_AVAILABLE and search_term:
                endpoint = ""
                if content_type == "Livres":
                    endpoint = f"{API_URL}/livres/?titre={search_term}"
                elif content_type == "Films":
                    endpoint = f"{API_URL}/films/?titre={search_term}"
                else:  # Musiques
                    endpoint = f"{API_URL}/musiques/?titre={search_term}"
                
                try:
                    response = requests.get(endpoint)
                    if response.status_code == 200:
                        results = response.json()
                except:
                    pass
            
            # Utilisation des modules de recommandation si l'API n'a pas donné de résultats
            if not results and MODULES_LOADED:
                try:
                    if content_type == "Livres":
                        results = recommandation.livres_recommandations(search_term)
                    elif content_type == "Films":
                        results = recommandation.films_recommandations(search_term)
                    else:  # Musiques
                        results = recommandation.musiques_recommandations(search_term)
                except:
                    pass
            
            # Fallback si ni l'API ni les modules n'ont donné de résultats
            #if not results and not df.empty:
            #    title_column = "titre" if "titre" in df.columns else "title" if "title" in df.columns else df.columns[0]
            #    sample_df = df.sample(min(5, len(df)))
            #    results = [{"titre": row[title_column]} for _, row in sample_df.iterrows()]
            
            # Construction de la réponse finale
            full_response = intro + "\n\n"
            
            if results:
                for i, item in enumerate(results[:5], 1):
                    full_response += f"{i}. {item['titre']}\n"
            else:
                full_response += "Désolé, je n'ai pas trouvé de recommandations correspondant à votre demande. Pourriez-vous préciser davantage ?"
 
            
            # Affichage de la réponse finale
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_message = f"Désolé, j'ai rencontré une erreur lors du traitement de votre demande : {str(e)}"
            message_placeholder.markdown(error_message)
            full_response = error_message
    
    # Ajout de la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info(
    """
    Ce dashboard a été créé pour faciliter l'interaction avec notre système de recommandation.
    Pour toute question ou suggestion, veuillez nous contacter.
    """
)