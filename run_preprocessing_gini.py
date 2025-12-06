"""
Script pour exécuter le prétraitement des données avec sélection de variables par Gini
Utilise preprocess_xgboost.py pour le prétraitement de base, puis ajoute la sélection par Gini

python run_preprocessing_gini.py
"""

import os
import sys
import pandas as pd

# Ajouter le dossier src au path
PROJECT_PATH = os.getcwd()
SRC_PATH = os.path.join(PROJECT_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from preprocess_xgboost import process_and_save_all as process_xgboost_base
from gini_selection import select_features_by_gini, load_gini_scores

def process_and_save_all_with_gini(project_path, windows=["FM12"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"], top_k=20):
    """
    Traite et sauvegarde tous les fichiers avec sélection de variables par Gini.
    Utilise preprocess_xgboost pour le prétraitement de base, puis ajoute la sélection par Gini.
    
    Args:
        project_path: Chemin du projet
        windows: Liste des fenêtres (ex: ["FM12"])
        segments: Liste des segments (ex: ["green", "red"])
        splits: Liste des splits (ex: ["train", "OOS", "OOT", "OOU"])
        top_k: Nombre de meilleures variables à garder (par défaut 20)
    """
    for window in windows:
        for segment in segments:
            # D'abord, traiter "train" pour calculer et sauvegarder les scores de Gini
            selected_features = None
            gini_scores = None
            
            # Traiter le split "train" en premier
            if "train" in splits:
                filename = f"train_{window[2:]}.csv"
                raw_path = os.path.join(project_path, "data", "raw", window, segment, filename)
                
                if os.path.exists(raw_path):
                    print(f"\n{'='*80}")
                    print(f"Traitement TRAIN avec sélection Gini : {raw_path}")
                    print(f"{'='*80}")
                    
                    # 1. Lire les données brutes
                    print("1. Lecture des données brutes...")
                    df = pd.read_csv(raw_path)
                    print(f"   Forme initiale : {df.shape}")
                    
                    # 2. Utiliser preprocess_xgboost pour le prétraitement de base
                    print("\n2. Prétraitement de base avec preprocess_xgboost...")
                    from preprocess_xgboost import preprocess_xgboost
                    df_processed = preprocess_xgboost(df, target_col='DFlag')
                    
                    # 3. Calculer les scores de Gini et sélectionner les meilleures variables
                    print("\n3. Sélection de variables par Gini...")
                    gini_scores_dir = os.path.join(project_path, "outputs", "gini_scores", window, segment)
                    gini_scores_path = os.path.join(gini_scores_dir, f"gini_scores_{window}.csv")
                    
                    df_final, gini_scores = select_features_by_gini(
                        df_processed, 
                        target_col='DFlag', 
                        min_gini_gain=0.0, 
                        top_k=top_k, 
                        save_scores_path=gini_scores_path
                    )
                    selected_features = df_final.columns.tolist()
                    
                    # 4. Sauvegarder les données prétraitées avec sélection
                    save_dir = os.path.join(project_path, "data", "processed", window, segment)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"train_{window[2:]}.csv")
                    df_final.to_csv(save_path, index=False)
                    print(f"\n✅ Données sauvegardées : {save_path}")
                    print(f"   Forme finale : {df_final.shape}")
                else:
                    print(f"⚠️  Fichier introuvable : {raw_path}")
            
            # Ensuite, traiter les autres splits en utilisant les mêmes variables que train
            for split in splits:
                if split == "train":
                    continue  # Déjà traité
                
                filename = f"{split}_{window[2:]}.csv"
                if split == "OOU":
                    filename = f"{split}.sas7bdat"
                
                raw_path = os.path.join(project_path, "data", "raw", window, segment, filename)
                
                if os.path.exists(raw_path):
                    print(f"\n{'='*80}")
                    print(f"Traitement {split} avec sélection Gini : {raw_path}")
                    print(f"{'='*80}")
                    
                    # Si train n'a pas été traité, essayer de charger les scores de Gini sauvegardés
                    if selected_features is None:
                        print("  Train non traité, tentative de chargement des scores de Gini...")
                        gini_scores = load_gini_scores(project_path, window=window, segment=segment)
                        if gini_scores is not None:
                            top_k_actual = min(top_k, len(gini_scores))
                            selected_features = gini_scores.head(top_k_actual)['feature'].tolist() + ['DFlag']
                            print(f"  ✓ Scores de Gini chargés, {len(selected_features)} variables sélectionnées")
                        else:
                            print("  ⚠️  Aucun score de Gini trouvé. Utilisation de toutes les variables.")
                    
                    # Utiliser preprocess_xgboost pour le prétraitement de base
                    print("\n1. Prétraitement de base avec preprocess_xgboost...")
                    from preprocess_xgboost import preprocess_xgboost
                    
                    # Créer le dossier de destination
                    save_dir = os.path.join(project_path, "data", "processed", window, segment)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{split}_{window[2:]}.csv")
                    
                    # Lire et traiter par chunks pour gérer les gros fichiers
                    chunk_size = 50000
                    first_chunk = True
                    chunk_num = 1
                    total_rows = 0
                    
                    # Lire le fichier selon son type
                    if split == "OOU":
                        # Traiter le fichier SAS
                        try:
                            import pyreadstat
                            print("  Lecture du fichier SAS avec pyreadstat...")
                            df, meta = pyreadstat.read_sas7bdat(raw_path)
                            print(f"  Fichier SAS lu: {len(df)} lignes, {len(df.columns)} colonnes")
                            
                            # Diviser en chunks pour le traitement
                            for i in range(0, len(df), chunk_size):
                                df_chunk = df.iloc[i:i + chunk_size].copy()
                                if len(df_chunk) == 0:
                                    break
                                
                                print(f"  Traitement du chunk {chunk_num} ({len(df_chunk)} lignes)...")
                                
                                # Prétraiter le chunk avec preprocess_xgboost
                                chunk_processed = preprocess_xgboost(df_chunk, target_col='DFlag')
                                
                                # Appliquer la sélection de variables si disponible
                                if selected_features is not None:
                                    available_features = [f for f in selected_features if f in chunk_processed.columns]
                                    chunk_final = chunk_processed[available_features]
                                else:
                                    chunk_final = chunk_processed
                                
                                # Sauvegarder
                                if first_chunk:
                                    chunk_final.to_csv(save_path, index=False, mode='w', header=True)
                                    first_chunk = False
                                else:
                                    chunk_final.to_csv(save_path, index=False, mode='a', header=False)
                                
                                total_rows += len(chunk_final)
                                chunk_num += 1
                                
                        except ImportError:
                            print("  ⚠️  pyreadstat n'est pas installé. Tentative avec pandas...")
                            df = pd.read_sas(raw_path, encoding="utf-8")
                            # Traiter normalement (sans chunks pour pandas)
                            df_processed = preprocess_xgboost(df, target_col='DFlag')
                            if selected_features is not None:
                                available_features = [f for f in selected_features if f in df_processed.columns]
                                df_final = df_processed[available_features]
                            else:
                                df_final = df_processed
                            df_final.to_csv(save_path, index=False)
                            total_rows = len(df_final)
                    else:
                        # Lire le fichier CSV par chunks
                        print("  Lecture et traitement par chunks (fichier volumineux)...")
                        
                        for chunk in pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False):
                            if len(chunk) == 0:
                                break
                            
                            print(f"  Traitement du chunk {chunk_num} ({len(chunk)} lignes)...")
                            
                            # Prétraiter le chunk avec preprocess_xgboost
                            chunk_processed = preprocess_xgboost(chunk, target_col='DFlag')
                            
                            # Appliquer la sélection de variables si disponible
                            if selected_features is not None:
                                available_features = [f for f in selected_features if f in chunk_processed.columns]
                                if chunk_num == 1 and len(available_features) < len(selected_features):
                                    missing_features = [f for f in selected_features if f not in chunk_processed.columns]
                                    print(f"  ⚠️  Variables manquantes: {missing_features}")
                                chunk_final = chunk_processed[available_features]
                            else:
                                chunk_final = chunk_processed
                            
                            # Sauvegarder
                            if first_chunk:
                                chunk_final.to_csv(save_path, index=False, mode='w', header=True)
                                first_chunk = False
                            else:
                                chunk_final.to_csv(save_path, index=False, mode='a', header=False)
                            
                            total_rows += len(chunk_final)
                            chunk_num += 1
                    
                    print(f"\n✅ Données sauvegardées : {save_path}")
                    print(f"   Total : {total_rows} lignes")
                else:
                    print(f"⚠️  Fichier introuvable : {raw_path}")

if __name__ == "__main__":
    print("="*80)
    print("PRÉTRAITEMENT DES DONNÉES AVEC SÉLECTION PAR GINI")
    print("(Utilise preprocess_xgboost.py pour le prétraitement de base)")
    print("="*80)
    print(f"Chemin du projet : {PROJECT_PATH}")
    print()
    
    # Prétraiter les données FM12 pour green et red
    # Le code utilisera preprocess_xgboost puis ajoutera la sélection par Gini
    process_and_save_all_with_gini(
        project_path=PROJECT_PATH,
        windows=["FM12"],  # Vous pouvez ajouter d'autres fenêtres: ["FM12", "FM24", "FM36", "FM48", "FM60"]
        segments=["green", "red"],
        splits=["train", "OOS", "OOT", "OOU"],
        top_k=20  # Nombre de meilleures variables à garder
    )
    
    print("\n" + "="*80)
    print("PRÉTRAITEMENT TERMINÉ")
    print("="*80)
    print("\nLes scores de Gini ont été sauvegardés dans:")
    print("  outputs/gini_scores/{window}/{segment}/gini_scores_{window}.csv")
    print("\nLes données prétraitées ont été sauvegardées dans:")
    print("  data/processed/{window}/{segment}/{split}_{window[2:]}.csv")
