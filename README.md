# Projet GIF-7005 – Équipe 12 (Session A25)

Ce dépôt contient le projet de session réalisé par l'équipe 12 dans le cadre du cours GIF-7005.

## Objectif du projet

À déterminer

## Structure du projet

GIF-7005/
├── notebooks/         # Notebooks Jupyter collaboratifs pour exploration et prototypage
├── src/               # Code Python modulaire (fonctions, classes, pipelines)
├── data/
│   ├── raw/           # Données brutes (non modifiées)
│   ├── processed/     # Données nettoyées et prêtes pour l'entraînement
├── models/            # Modèles entraînés (.pkl, .pt, etc.)
├── outputs/           # Graphiques, résultats, logs d'expériences
├── environment.yml    # Dépendances Conda pour reproduire l'environnement
├── README.md          # Documentation du projet

## Démarrage rapide

### Option 1 : avec Conda

1. Cloner le dépôt :

   git clone https://github.com/GIF-7005-Equipe-12/Projet.git
   cd Projet

2. Créer l’environnement Conda :

   conda env create -f environment.yml
   conda activate gif7005-env

3. Synchroniser les notebooks avec Jupytext :

   # Pour activer la synchronisation entre .ipynb et .py
   jupytext --set-formats ipynb,py notebooks/nom_du_notebook.ipynb

### Option 2 : avec un environnement virtuel Python (venv + pip)

1. Cloner le dépôt :

   git clone https://github.com/GIF-7005-Equipe-12/Projet.git
   cd Projet

2. Créer et activer l’environnement virtuel :

   python -m venv venv

   # Sur Windows :
   venv\Scripts\activate

   # Sur macOS/Linux :
   source venv/bin/activate

3. Installer les dépendances :

   pip install -r requirements.txt

4. Synchroniser les notebooks avec Jupytext :

   jupytext --set-formats ipynb,py notebooks/nom_du_notebook.ipynb

## Données

Les données brutes sont stockées dans `data/raw/`. Les données nettoyées sont dans `data/processed/`.

Les fichiers volumineux ne sont pas versionnés dans GitHub. 

## Modèles

Les modèles entraînés sont sauvegardés dans `models/`. 

## Collaboration

- Utiliser des branches pour chaque fonctionnalité (`feature/nom`)
- Faire des pull requests pour intégrer du code
- Synchroniser les notebooks avec des fichiers `.py` via Jupytext
- Nettoyer les outputs avec nbstripout avant de committer

## Convention de commit des notebooks avec Jupytext

Dans ce projet, les notebooks Jupyter sont synchronisés avec des fichiers `.py` grâce à Jupytext. Pour faciliter la lecture, le versionnage et la collaboration.

### Étapes pour travailler avec un notebook

1. Créer ou ouvrir un notebook `.ipynb` dans `notebooks/`.

2. Synchroniser le notebook avec un fichier `.py` :

   jupytext --set-formats ipynb,py notebooks/mon_notebook.ipynb

   Cela crée automatiquement `mon_notebook.py` dans le même dossier.

3. Travailler dans le notebook comme d’habitude.

4. Ajouter les changements:

   git add .

5. Committer avec un message clair :

   git commit -m "Ajout du notebook d'exploration synchronisé avec Jupytext"

### Pourquoi cette convention ?

- Les fichiers `.py` sont lisibles et fusionnables dans Git.
- Les fichiers `.ipynb` sont verbeux et difficiles à suivre en collaboration.
