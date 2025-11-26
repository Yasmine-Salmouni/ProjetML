"""
Script pour exécuter le prétraitement des données pour XGBoost

python run_preprocessing.py
"""

import os
import sys

# Ajouter le dossier src au path
PROJECT_PATH = os.getcwd()
SRC_PATH = os.path.join(PROJECT_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from preprocess_xgboost import process_and_save_all

if __name__ == "__main__":
    print("="*80)
    print("PRÉTRAITEMENT DES DONNÉES POUR XGBOOST")
    print("="*80)
    print(f"Chemin du projet : {PROJECT_PATH}")
    print()
    
    # Prétraiter les données FM12, FM36 et FM60 pour green et red
    process_and_save_all(
        project_path=PROJECT_PATH,
        windows=["FM12", "FM36", "FM60"],
        segments=["green", "red"],
        splits=["train", "OOS", "OOT", "OOU"]
    )
    
    print("\n" + "="*80)
    print("PRÉTRAITEMENT TERMINÉ")
    print("="*80)

