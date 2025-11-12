# **Projet GIF-7005 – Équipe 12 (Session A25)**
![Pipeline](https://img.shields.io/badge/pipeline-passed-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/AlisonOuellet/GenomicResistancePreditor?style=flat-square)

Ce dépôt contient le projet de session réalisé par l'équipe 12 dans le cadre du cours GIF-7005 (GIF-4005) à l'Université Laval.

---

### Table of Contents
1. [Objectif](#Objectif)
2. [Structure](#Structure)
3. [Démarrage](#Démarrage)
4. [Données](#Données)
5. [Modèles](#Modèles)
6. [Collaboration](#collaboration)
7. [Convention de commit des notebooks avec Jupytext](#notebook)

---

## Objectif

À déterminer

---

## Structure
```
GIF-7005/
├── /         # Notebooks Jupyter collaboratifs pour exploration et prototypage
├── src/               # Code Python modulaire (fonctions, classes, pipelines)
├── data/
│   ├── raw/           # Données brutes (non modifiées)
│   └── processed/     # Données nettoyées et prêtes pour l'entraînement
├── models/            # Modèles entraînés (.pkl, .pt, etc.)
├── outputs/           # Graphiques, résultats, logs d'expériences
├── environment.yml    # Dépendances Conda pour reproduire l'environnement
└── README.md          # Documentation du projet
```
---
## Démarrage

### Option 1 : installation avec Conda

1. Cloner le dépôt :
   git clone https://github.com/GIF-7005-Equipe-12/Projet.git
   cd Projet

2. Créer et activer l’environnement Conda :
   conda env create -f environment.yml
   conda activate gif7005-env

3. Synchroniser les notebooks avec Jupytext :

   # Pour activer la synchronisation entre .ipynb et .py
   jupytext --set-formats ipynb,py /nom_du_notebook.ipynb

4. Télécharger les données FMDataV2.zip et OutOfUniverse.zip
> Les données sont disponibles ici: https://data.mendeley.com/datasets/bzr2rxttvz/3
5. Extraire les données **CSV** FMDataV2 dans le dossier /data/raw
```
Exemple: 
/data/raw/FM12
/data/raw/FM24
/data/raw/FM36
/data/raw/FM48
/data/raw/FM60
```
6. Extraire les données OutOfUniverse.zip dans le dossier /data/raw/FM12 et **renommer les fichiers**.
> Il y a seulement des données OOU pour la fenêtre FM12.
```
Exemple:
/data/raw/FM12/green/OOU.sas7bdat
/data/raw/FM12/red/OOU.sas7bdat
```

### Option 2 : installation avec un environnement virtuel Python (venv + pip)

1. Cloner le dépôt :
   git clone https://github.com/GIF-7005-Equipe-12/Projet.git
   cd Projet

2. Créer l’environnement virtuel :
   python -m venv venv

3. Activer l'environnement
   # Sous Windows :
   venv\Scripts\activate

   # Sous macOS/Linux :
   source venv/bin/activate

3. Installer les dépendances :
   pip install -r requirements.txt

4. Synchroniser les notebooks avec Jupytext :

   jupytext --set-formats ipynb,py /nom_du_notebook.ipynb

5. Télécharger les données FMDataV2.zip et OutOfUniverse.zip
> Les données sont disponibles ici: https://data.mendeley.com/datasets/bzr2rxttvz/3
6. Extraire les données **CSV** FMDataV2 dans le dossier /data/raw
```
Exemple: 
/data/raw/FM12
/data/raw/FM24
/data/raw/FM36
/data/raw/FM48
/data/raw/FM60
```
7. Extraire les données OutOfUniverse.zip dans le dossier /data/raw/FM12 et **renommer les fichiers**.
> Il y a seulement des données OOU pour la fenêtre FM12.
```
Exemple:
/data/raw/FM12/green/OOU.sas7bdat
/data/raw/FM12/red/OOU.sas7bdat
```

---

## Données
> Les données sont disponibles ici: https://data.mendeley.com/datasets/bzr2rxttvz/3
Les données brutes sont stockées dans `data/raw/`. Les données nettoyées sont dans `data/processed/`.

Les fichiers volumineux ne sont pas versionnés dans GitHub. 

---
## Modèles

Les modèles entraînés sont sauvegardés dans `models/`. 

---

## Collaboration

- Utiliser des branches pour chaque fonctionnalité (`feature/nom`)
- Faire des pull requests pour intégrer du code
- Synchroniser les notebooks avec des fichiers `.py` via Jupytext
- Nettoyer les outputs avec nbstripout avant de committer

---

## Convention de commit des notebooks avec Jupytext

Dans ce projet, les notebooks Jupyter sont synchronisés avec des fichiers `.py` grâce à Jupytext. Pour faciliter la lecture, le versionnage et la collaboration.

### Étapes pour travailler avec un notebook

1. Créer ou ouvrir un notebook `.ipynb` dans `/`.

2. Synchroniser le notebook avec un fichier `.py` :

   jupytext --set-formats ipynb,py project.ipynb

   Cela crée automatiquement `project.py` dans le même dossier.

3. Travailler dans le notebook comme d’habitude.

4. Ajouter les changements:

   git add .

5. Committer avec un message clair :

   git commit -m "Ajout du notebook d'exploration synchronisé avec Jupytext"

### Pourquoi cette convention ?

- Les fichiers `.py` sont lisibles et fusionnables dans Git.
- Les fichiers `.ipynb` sont verbeux et difficiles à suivre en collaboration.
