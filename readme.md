# Utilisation de l'application

### 1. Installation des packages
Creer un dossier d'envrionnement, en tapant à la racine du projet :
    - Pour Windows : python -m venv venv
    - Pour Linux : python3 -m venv venv
    - Installer les packages : pip install -r requirements.txt

### 2. Collecte des données via APIs publics
Executer le fichier de collecte des données
    - Avoir un compte sur themoviedb
    - Copier la clé de l'API public à remplacer dans la variables gobale du fichier modules/collect_data
    - Se deplacer dans le dossier en faisant : cd modules
    - Lancer le fichier en tapant : python collect_data.py

### 3. Lancer les APIs
Cette partie est dependante de la partie 1, s'il n'ya pas de contenu dans le dossier data ou si on veut faire une nouvelle collecte

Activer le fichier venv en tapant : 
    - Sur Windows : venv\Scripts\activate
    - Sur Linux : source venv/bin/activate

Taper ensuite cette commande pour lancer lancer l'API : uvicorn api.main:app --reload

Lien de l'API : htt://localhost:8000
Lien de docs l'API : htt://localhost:8000/docs


