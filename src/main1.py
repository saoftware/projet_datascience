import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# Configuration de la page
st.set_page_config(
    page_title="Chatbot Culture & Loisirs",
    page_icon="🎬",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e1b4b 0%, #7c3aed 100%);
    }
    .stChatMessage {
        background-color: rgba(30, 27, 75, 0.6);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #475569;
    }
    h1 {
        color: #e0e7ff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7c3aed 0%, #ec4899 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
# --- 1. Chargement du modèle ---
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# --- 2. Chargement des données ---
@st.cache_data
def load_data():
    try:
        films = pd.read_csv('data/data_cleaned/films.csv')
        musiques = pd.read_csv('data/data_cleaned/musiques.csv')
        livres = pd.read_csv('data/data_cleaned/livres.csv')
        return films, musiques, livres
    except FileNotFoundError:
        st.error("❌ CSV introuvables.")
        return None, None, None

# --- 3. Création des embeddings + index FAISS ---
@st.cache_data
def create_faiss_index(_model, films, musiques, livres):
    indices = {}
    embeddings = {}
    categories = {
        'films': (films, ['titre', 'genre', 'description']),
        'musiques': (musiques, ['titre', 'artiste', 'genre', 'album']),
        'livres': (livres, ['titre', 'auteur', 'genre', 'description'])
    }

    for cat, (df, cols) in categories.items():
        df['search_text'] = df.apply(
            lambda x: " ".join([str(x.get(c, "")) for c in cols]), axis=1
        )
        emb = _model.encode(df['search_text'].tolist(), show_progress_bar=False)
        faiss.normalize_L2(emb)  # cosinus
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        indices[cat] = index
        embeddings[cat] = emb

    return indices, embeddings

# Fonction de recherche sémantique
def search_recommendations(query, model, category, films, musiques, livres, indices, top_k=3):
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)
    results = []

    cats = ['films', 'musiques', 'livres'] if category == 'all' else [category]

    for cat in cats:
        df = {'films': films, 'musiques': musiques, 'livres': livres}[cat]
        if df is None:
            continue
        scores, idx = indices[cat].search(query_emb, top_k)
        for score, i in zip(scores[0], idx[0]):
            if score > 0.2:
                row = df.iloc[i]
                results.append({
                    'type': {'films': '🎬 Film', 'musiques': '🎵 Musique', 'livres': '📚 Livre'}[cat],
                    'titre': row['titre'],
                    'details': f"**Genre:** {row.get('genre', 'N/A')} | **Année:** {row.get('annee', 'N/A')}",
                    'description': str(row.get('description', ''))[:200] + '...',
                    'score': float(score)
                })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

# Interface principale
def main():
    st.title("🎬 Chatbot Culture & Loisirs")
    st.markdown("### Votre assistant intelligent de récommandations de livres, de films et de musiques")

    # 1️⃣ Chargement des données et du modèle
    model = load_model()
    films, musiques, livres = load_data()
    if films is None:
        st.stop()

    # 2️⃣ Création des index FAISS (avant tout appel)
    indices, embeddings = create_faiss_index(model, films, musiques, livres)

    # 3️⃣ Sidebar : filtres et statistiques
    with st.sidebar:
        st.header("⚙️ Paramètres")
        category = st.selectbox(
            "Catégorie",
            ['all', 'films', 'musiques', 'livres'],
            format_func=lambda x: {
                'all': '🌟 Toutes',
                'films': '🎬 Films',
                'musiques': '🎵 Musiques',
                'livres': '📚 Livres'
            }[x]
        )
        st.markdown("---")
        st.metric("Films", len(films))
        st.metric("Musiques", len(musiques))
        st.metric("Livres", len(livres))
        st.markdown("---")
        st.markdown("### 💡 Exemples")
        st.code("fantasy épique")
        st.code("musique dance 2019")
        st.code("thriller suspense")

    # 4️⃣ Historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour ! 👋 Que recherchez-vous ?"}
        ]

    # 5️⃣ Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "recommendations" in message:
                for rec in message["recommendations"]:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{rec['type']} : {rec['titre']}</h4>
                        <p>{rec['details']}</p>
                        <p style='color:#94a3b8;font-size:0.9em;'>{rec['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # 6️⃣ Input utilisateur + recherche
    if prompt := st.chat_input("Recherchez un film, musique ou livre..."):
        # Ajouter message user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Recommandations
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche en cours..."):
                recommendations = search_recommendations(
                    prompt, model, category, films, musiques, livres, indices
                )
                if recommendations:
                    response = f"J'ai trouvé {len(recommendations)} recommandation(s) :"
                    st.markdown(response)
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{rec['type']} : {rec['titre']}</h4>
                            <p>{rec['details']}</p>
                            <p style='color:#94a3b8;font-size:0.9em;'>{rec['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "recommendations": recommendations
                    })
                else:
                    response = "😔 Aucun résultat, essayez d'autres mots-clés."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()