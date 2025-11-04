# api/main.py - Version Hybride (Règles + Hugging Face)
import os
import sys
from typing import List, Optional
import re

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Hugging Face
try:
    from transformers import pipeline, set_seed
    HF_AVAILABLE = True
    print("🤗 Transformers disponible")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Transformers non installé (pip install transformers)")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import data_cleaning, config

# ------------------------------------------------------------------
# Initialisation FastAPI
# ------------------------------------------------------------------
app = FastAPI(
    title="Chatbot Culture Hybride",
    version="3.0.0",
    description="Chatbot intelligent combinant règles et IA générative",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Chargement du modèle Hugging Face (optionnel)
# ------------------------------------------------------------------
llm_generator = None

if HF_AVAILABLE:
    try:
        print("📥 Chargement du modèle GPT-2...")
        llm_generator = pipeline(
            "text-generation",
            model="gpt2",  # ou "gpt2-medium" (meilleur)
            max_length=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=50256
        )
        set_seed(42)
        print("✅ GPT-2 chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement GPT-2: {e}")
        llm_generator = None

# ------------------------------------------------------------------
# Chargement des données
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def _csv_path(name: str) -> str:
    return os.path.join(BASE_DIR, "data", "cleaned", f"{name}.csv")

for f in ("films", "livres", "musiques"):
    if not os.path.exists(_csv_path(f)):
        print(f"Fichier {_csv_path(f)} manquant : nettoyage...")
        data_cleaning.load_clean_and_save_data()
        break

films_df = config.import_data(_csv_path("films"))
livres_df = config.import_data(_csv_path("livres"))
musiques_df = config.import_data(_csv_path("musiques"))

print(f"✅ Données: {len(films_df)} films, {len(livres_df)} livres, {len(musiques_df)} musiques")

# ------------------------------------------------------------------
# Modèles Pydantic
# ------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    category: str = "general"
    conversation_history: Optional[List[ChatMessage]] = []

# ------------------------------------------------------------------
# Fonctions de recherche (inchangées)
# ------------------------------------------------------------------
def _search_df(df: pd.DataFrame, query: str, cols: List[str], limit: int = 5):
    mask = pd.Series(False, index=df.index)
    query_lower = query.lower()
    
    for c in cols:
        if c in df.columns:
            mask |= df[c].astype(str).str.lower().str.contains(query_lower, na=False, regex=False)
    
    return df[mask].head(limit)

def extract_keywords(message: str) -> List[str]:
    stop_words = {
        "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
        "le", "la", "les", "un", "une", "des", "de", "du",
        "et", "ou", "mais", "donc", "car", "pour", "sur", "dans",
        "peux", "veux", "cherche", "trouve", "recommande", "suggère",
        "livre", "film", "musique", "chanson", "album", "auteur", "réalisateur",
        "me", "te", "se", "qui", "que", "quoi", "comment", "ai", "as", "est"
    }
    
    words = re.findall(r'\b\w+\b', message.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords

def search_items(query: str, category: str, limit: int = 5):
    category = category.lower()
    keywords = extract_keywords(query)
    
    if not keywords:
        keywords = [query.lower()]
    
    search_query = " ".join(keywords)
    
    if category == "films":
        df = films_df
        hits = _search_df(df, search_query, ["titre", "director", "description", "genre"])
        return [
            {
                "type": "Film",
                "titre": row.titre,
                "details": f"🎬 Réalisateur: {row.director} | 📅 {row.year}",
                "description": str(row.description)[:200] + "..." if len(str(row.description)) > 200 else str(row.description),
            }
            for _, row in hits.iterrows()
        ]
    
    elif category == "livres":
        df = livres_df
        hits = _search_df(df, search_query, ["titre", "auteur", "description", "genre"])
        return [
            {
                "type": "Livre",
                "titre": row.titre,
                "details": f"✍️ Auteur: {row.auteur} | 📅 {row.annee}",
                "description": str(row.description)[:200] + "..." if len(str(row.description)) > 200 else str(row.description),
            }
            for _, row in hits.iterrows()
        ]
    
    elif category == "musiques":
        df = musiques_df
        hits = _search_df(df, search_query, ["titre", "artist", "album", "genre"])
        return [
            {
                "type": "Musique",
                "titre": row.titre,
                "details": f"🎤 Artiste: {row.artist} | 💿 Album: {row.album}",
                "description": f"Année: {row.year}" if hasattr(row, 'year') else "",
            }
            for _, row in hits.iterrows()
        ]
    
    return []

# ------------------------------------------------------------------
# NOUVEAU : Génération avec modèle HF
# ------------------------------------------------------------------
def generate_with_llm(prompt: str, max_tokens: int = 80) -> Optional[str]:
    """Génère une réponse avec GPT-2"""
    if not llm_generator:
        return None
    
    try:
        result = llm_generator(
            prompt,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            truncation=True
        )
        
        # Nettoyer la réponse
        response = result[0]['generated_text']
        response = response.replace(prompt, "").strip()
        
        # Limiter à 2-3 phrases
        sentences = response.split('.')
        response = '. '.join(sentences[:3]).strip()
        if response and not response.endswith('.'):
            response += '.'
        
        return response
    
    except Exception as e:
        print(f"❌ Erreur génération LLM: {e}")
        return None

# ------------------------------------------------------------------
# NOUVEAU : Réponses hybrides (Règles + LLM)
# ------------------------------------------------------------------
def generate_conversational_response(
    message: str, 
    items: List[dict], 
    category: str, 
    history: List[ChatMessage],
    use_llm: bool = True
) -> str:
    """Génère une réponse en combinant règles et IA"""
    
    message_lower = message.lower()
    
    # ============================================================
    # PARTIE 1 : RÈGLES (pour les cas simples et prédictibles)
    # ============================================================
    
    # 1. Salutations
    greetings = ["bonjour", "salut", "hello", "hey", "coucou", "bonsoir"]
    if any(g in message_lower for g in greetings) and len(message.split()) <= 3:
        return "Bonjour ! Je suis ravi de t'aider à découvrir de super contenus culturels. Tu cherches un film, un livre ou de la musique ? 😊"
    
    # 2. Remerciements
    thanks = ["merci", "thank", "super", "génial", "parfait", "ok", "bien", "d'accord"]
    if any(t in message_lower for t in thanks) and len(message.split()) <= 4:
        import random
        responses = [
            "Avec plaisir ! N'hésite pas si tu as besoin d'autres recommandations 😊",
            "Content d'avoir pu t'aider ! Autre chose ?",
            "De rien ! Je suis là si tu veux d'autres suggestions 🎬📚🎵"
        ]
        return random.choice(responses)
    
    # 3. Recommandations avec résultats (RÈGLES pour garantir la cohérence)
    if items:
        category_emoji = {"films": "🎬", "livres": "📚", "musiques": "🎵"}
        emoji = category_emoji.get(category, "✨")
        
        if len(items) == 1:
            intro = f"{emoji} J'ai trouvé exactement ce qu'il te faut !"
        elif len(items) <= 3:
            intro = f"{emoji} Voici {len(items)} {category} qui devraient te plaire !"
        else:
            intro = f"{emoji} Super ! J'ai {len(items)} excellentes suggestions pour toi !"
        
        keywords = extract_keywords(message)
        if keywords:
            context = f"Basé sur ta recherche '{' '.join(keywords[:2])}', voici mes recommandations :"
        else:
            context = "Voici ce que j'ai sélectionné pour toi :"
        
        return f"{intro}\n\n{context}"
    
    # 4. Aucun résultat (RÈGLES avec suggestions précises)
    if not items and any(kw in message_lower for kw in ["cherche", "trouve", "veux", "recommand"]):
        keywords = extract_keywords(message)
        suggestions = {
            "films": ["science-fiction", "action", "comédie", "thriller"],
            "livres": ["roman", "science-fiction", "policier", "fantastique"],
            "musiques": ["rock", "pop", "jazz", "électronique"]
        }
        
        if keywords:
            return f"🤔 Hmm, je n'ai rien trouvé pour '{' '.join(keywords[:2])}'...\n\nEssaie d'être plus précis ! Par exemple : un titre, un auteur, ou un genre comme {', '.join(suggestions.get(category, ['action', 'comédie'])[:3])} ?"
        else:
            return f"😅 Je n'ai pas bien compris ta recherche.\n\nDis-moi ce que tu cherches : un titre, un auteur, ou un genre comme {', '.join(suggestions.get(category, ['action'])[:2])} ?"
    
    # ============================================================
    # PARTIE 2 : LLM (pour les conversations libres et complexes)
    # ============================================================
    
    if use_llm and llm_generator:
        # Questions sur l'assistant
        if any(q in message_lower for q in ["qui es", "tu es", "c'est quoi", "qu'est-ce"]):
            prompt = "tu es un assistant spécialisé dans la recommandation de films, de livres et de music. dites-moi ce que vous recherchez!"
            llm_response = generate_with_llm(prompt, max_tokens=60)
            if llm_response:
                return f"🤖 {llm_response}"
        
        # Questions "comment ça va"
        if "comment" in message_lower and ("vas" in message_lower or "alles" in message_lower):
            prompt = "Someone asks how you are. Respond briefly and enthusiastically, then ask what they're looking for today (movies, books, music). Keep it under 2 sentences:\n\nResponse:"
            llm_response = generate_with_llm(prompt, max_tokens=50)
            if llm_response:
                return f"😊 {llm_response}"
        
        # Questions ouvertes / conseils généraux
        if any(word in message_lower for word in ["pourquoi", "comment", "conseil", "avis", "opinion", "penses"]):
            # Créer un contexte basé sur la catégorie
            context_map = {
                "films": "movies and cinema",
                "livres": "books and literature",
                "musiques": "music and songs"
            }
            context = context_map.get(category, "cultural content")
            
            prompt = f"You are a cultural expert. Answer this question about {context} in a friendly, concise way (2-3 sentences max):\n\nQuestion: {message}\n\nAnswer:"
            llm_response = generate_with_llm(prompt, max_tokens=80)
            if llm_response:
                return f"💭 {llm_response}"
    
    # ============================================================
    # FALLBACK : Règle par défaut
    # ============================================================
    return "🤔 Je ne suis pas sûr de comprendre... Tu peux me dire ce que tu cherches ? Par exemple : 'Je veux un film d'action' ou 'Trouve-moi un livre de science-fiction' !"

# ------------------------------------------------------------------
# Route principale
# ------------------------------------------------------------------
@app.post("/chat", tags=["Chatbot"])
async def chat_with_bot(data: ChatRequest):
    """Chatbot hybride : règles + IA générative"""
    try:
        message = data.message.strip()
        category = data.category.lower()
        history = data.conversation_history or []
        
        # Détection d'intention de recherche
        search_keywords = [
            "cherche", "trouve", "recommand", "suggère", "propose", "veux", "voudrais",
            "conseil", "idée", "parle", "histoire", "genre", "style", "comme"
        ]
        is_search_request = any(kw in message.lower() for kw in search_keywords)
        
        # Recherche dans la base si nécessaire
        items = []
        if is_search_request and category in ["films", "livres", "musiques"]:
            items = search_items(message, category, limit=5)
        
        # Générer la réponse (hybride)
        response = generate_conversational_response(
            message, items, category, history, use_llm=True
        )
        
        return {
            "response": response,
            "type": "search_results" if items else "conversation",
            "items": items,
            "used_llm": llm_generator is not None
        }
    
    except Exception as e:
        print(f"❌ Erreur /chat: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "response": "😅 Oups, un problème technique. Peux-tu reformuler ?",
            "type": "error",
            "items": []
        }

# ------------------------------------------------------------------
# Routes existantes (inchangées)
# ------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health():
    return {
        "message": "API Hybride opérationnelle ✅",
        "llm_status": "✅ Actif" if llm_generator else "❌ Inactif",
        "stats": {
            "films": len(films_df),
            "livres": len(livres_df),
            "musiques": len(musiques_df)
        }
    }

@app.get("/films", tags=["Catalogue"])
async def get_films(titre: str = Query(None)):
    if titre:
        return search_items(titre, "films")
    return films_df.sample(min(10, len(films_df))).to_dict(orient="records")

@app.get("/livres", tags=["Catalogue"])
async def get_livres(titre: str = Query(None)):
    if titre:
        return search_items(titre, "livres")
    return livres_df.sample(min(10, len(livres_df))).to_dict(orient="records")

@app.get("/musiques", tags=["Catalogue"])
async def get_musiques(titre: str = Query(None)):
    if titre:
        return search_items(titre, "musiques")
    return musiques_df.sample(min(10, len(musiques_df))).to_dict(orient="records")

@app.get("/stats", tags=["Debug"])
async def get_stats():
    return {
        "films": {
            "count": len(films_df),
            "columns": list(films_df.columns),
            "sample": films_df["titre"].head(3).tolist() if "titre" in films_df.columns else []
        },
        "livres": {
            "count": len(livres_df),
            "columns": list(livres_df.columns),
            "sample": livres_df["titre"].head(3).tolist() if "titre" in livres_df.columns else []
        },
        "musiques": {
            "count": len(musiques_df),
            "columns": list(musiques_df.columns),
            "sample": musiques_df["titre"].head(3).tolist() if "titre" in musiques_df.columns else []
        },
        "llm_active": llm_generator is not None
    }