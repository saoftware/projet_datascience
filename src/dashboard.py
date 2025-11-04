import streamlit as st
import pandas as pd
import requests
import os
import sys

# Ajout du chemin du projet
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import des modules
try:
    from modules import recommandation, config, data_cleaning
    MODULES_LOADED = True
    print("✅ Modules chargés avec succès!")
except ImportError as e:
    print(f"⚠️ Erreur lors de l'importation des modules: {e}")
    MODULES_LOADED = False

# Configuration
API_URL = "http://127.0.0.1:8000"

# Vérification API
def is_api_available():
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

API_AVAILABLE = is_api_available()

# Configuration Streamlit
st.set_page_config(
    page_title="Assistant Culturel 🎬📚🎵",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .recommendation-card h4 {
        margin: 0 0 10px 0;
        font-size: 1.2em;
    }
    .recommendation-card p {
        margin: 5px 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Paramètres")
st.sidebar.markdown("---")

content_type = st.sidebar.selectbox(
    "📂 Catégorie",
    ["Livres", "Films", "Musiques"],
    help="Sélectionnez le type de contenu pour les recommandations"
)

# Indicateur de statut API
if API_AVAILABLE:
    st.sidebar.success("✅ API connectée")
else:
    st.sidebar.error("❌ API déconnectée")
    st.sidebar.caption("Lancez : `uvicorn api.main:app --reload --port 8000`")

# Bouton pour réinitialiser la conversation
if st.sidebar.button("🔄 Nouvelle conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Astuces d'utilisation:**
- Dites "Bonjour" pour commencer
- Demandez un film, livre ou musique; ex:"je veux un film de science-fiction"... 
- NB:L'assistant est toujours en déveleppement
-Donc tous les films, livres et musiques ne s'y trouvent pas pour le moment! Merci
""")

# Chargement des données (optionnel pour l'aperçu)
@st.cache_data
def load_data(content_type):
    try:
        if MODULES_LOADED:
            if content_type == "Livres":
                return recommandation.df_livres
            elif content_type == "Films":
                return recommandation.df_films
            else:
                return recommandation.df_musiques
        else:
            path = f"data/cleaned/{content_type.lower()}.csv"
            return pd.read_csv(path)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Données {content_type} non disponibles")
        return pd.DataFrame()

df = load_data(content_type)

# En-tête principal
st.title("🤖 Assistant Culturel Intelligent")
st.markdown(f"""
Bienvenue ! Je suis ton assistant pour découvrir des **{content_type.lower()}** 
qui correspondent à tes goûts. Pose-moi des questions, demande des recommandations !
""")

# Statistiques rapides (collapsable)
if not df.empty:
    with st.expander(f"📊 Aperçu de la base de données ({content_type})"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(df))
        with col2:
            if "titre" in df.columns:
                st.metric("Titres uniques", df["titre"].nunique())
        with col3:
            if content_type == "Livres" and "auteur" in df.columns:
                st.metric("Auteurs", df["auteur"].nunique())
            elif content_type == "Films" and "director" in df.columns:
                st.metric("Réalisateurs", df["director"].nunique())
            elif content_type == "Musiques" and "artist" in df.columns:
                st.metric("Artistes", df["artist"].nunique())
        
        st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# ========================================
# SECTION CHATBOT PRINCIPAL
# ========================================

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Salut ! 👋 Dis-moi ce que tu cherches et je te trouve les meilleurs contenus culturels !",
            "recommendations": []
        }
    ]

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        # Afficher les recommandations avec un meilleur design
        if "recommendations" in message and message["recommendations"]:
            for rec in message["recommendations"]:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{rec.get('type', '')} : {rec.get('titre', 'Sans titre')}</h4>
                    <p><strong>{rec.get('details', '')}</strong></p>
                    <p style='opacity: 0.9;'>{rec.get('description', '')}</p>
                </div>
                """, unsafe_allow_html=True)

# Input utilisateur
if prompt := st.chat_input("💬 Tape ton message ici..."):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Afficher le message utilisateur immédiatement
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Réponse de l'assistant
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Réflexion en cours..."):
            try:
                if not API_AVAILABLE:
                    st.error("🔌 L'API n'est pas disponible. Impossible de répondre.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "❌ API déconnectée. Lance FastAPI avec : `uvicorn api.main:app --reload --port 8000`",
                        "recommendations": []
                    })
                else:
                    # Préparer l'historique pour l'API (10 derniers messages max)
                    history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[-10:]
                        if msg["role"] in ["user", "assistant"]
                    ]
                    
                    # Appel à l'API
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": prompt,
                            "category": content_type.lower(),
                            "conversation_history": history
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Afficher la réponse textuelle
                        st.markdown(data["response"])
                        
                        # Afficher les recommandations
                        items = data.get("items", [])
                        if items:
                            for item in items:
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>{item.get('type', '')} : {item.get('titre', 'Sans titre')}</h4>
                                    <p><strong>{item.get('details', '')}</strong></p>
                                    <p style='opacity: 0.9;'>{item.get('description', '')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Sauvegarder dans l'historique
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["response"],
                            "recommendations": items
                        })
                    
                    else:
                        error_msg = f"❌ Erreur API (code {response.status_code})"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "recommendations": []
                        })
            
            except requests.exceptions.Timeout:
                error_msg = "⏱️ L'API met trop de temps à répondre. Réessaie !"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "recommendations": []
                })
            
            except requests.exceptions.ConnectionError:
                error_msg = "🔌 Impossible de se connecter à l'API. Vérifie qu'elle est lancée."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "recommendations": []
                })
            
            except Exception as e:
                error_msg = f"💥 Erreur inattendue : {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "recommendations": []
                })

# Footer
st.markdown("---")
st.caption("🤖 Propulsé par FastAPI + Streamlit | Assistant culturel v2.0")