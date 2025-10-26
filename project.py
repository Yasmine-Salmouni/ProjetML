# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: gif7005-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Projet

# %% [markdown]
# ## Prérequis

# %%
# Modules standards
import sys
import os
import importlib
from ydata_profiling import ProfileReport
import pandas as pd

# Définir les chemins du projet
PROJECT_PATH = os.getcwd()
SRC_PATH = os.path.join(PROJECT_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

print("Chemin du projet :", PROJECT_PATH)
print("Chemin du dossier src :", SRC_PATH)

# Importation des modules personnalisés
import explore_data
import preprocess

# Rechargement si modification en cours de session
importlib.reload(explore_data)
importlib.reload(preprocess)

from explore_data import * 
from preprocess import * 

# %% [markdown]
# ## Prétraitement des données

# %%
process_and_save_all(PROJECT_PATH, windows=["FM12"], segments=["red"])

# %% [markdown]
# ## Importation des données prétraitées

# %%
data_train = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=["train"])
X_train, y_train = data_train.drop(columns=['DFlag']),data_train['DFlag']
data_test = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=["OOS"])
X_test, y_test = data_test.drop(columns=['DFlag']), data_test['DFlag']

# %% [markdown]
# ## Exploration des données

# %%
from explore_data import *

save_path = os.path.join(PROJECT_PATH, "outputs", "exploration", "rapport_FM12.html")

data_to_explore = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'])
summarize_data_to_html(data_to_explore, "FM12 - Rapport", save_path)
