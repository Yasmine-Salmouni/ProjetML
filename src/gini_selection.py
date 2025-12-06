"""
Script de sélection de caractéristiques basée sur l'impureté de Gini
Utilise l'importance Gini d'un modèle XGBoost pour sélectionner les caractéristiques les plus pertinentes
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from preprocess_xgboost import load_processed_data, preprocess_xgboost, read_sas_file


def calculate_gini_importance(data, target_col='DFlag', method='xgboost', 
                              n_estimators=100, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, sample_size=None):
    """
    Calcule l'importance Gini des caractéristiques en utilisant un modèle d'arbre de décision.
    Utilise les mêmes paramètres que train_xgboost.py pour garantir la cohérence.
    
    Args:
        data: DataFrame avec les caractéristiques et la variable cible
        target_col: Nom de la colonne cible (par défaut 'DFlag')
        method: Méthode à utiliser ('xgboost' ou 'decision_tree')
        n_estimators: Nombre d'arbres pour XGBoost (par défaut 100, comme train_xgboost.py)
        max_depth: Profondeur maximale des arbres (par défaut 6, comme train_xgboost.py)
        learning_rate: Taux d'apprentissage (par défaut 0.1, comme train_xgboost.py)
        subsample: Proportion d'échantillons utilisés (par défaut 0.8, comme train_xgboost.py)
        colsample_bytree: Proportion de caractéristiques utilisées (par défaut 0.8, comme train_xgboost.py)
        sample_size: Nombre d'échantillons à utiliser (None = utiliser toutes les données)
    
    Returns:
        Tuple (DataFrame avec les caractéristiques et leur importance Gini, modèle entraîné)
    """
    print("="*80)
    print("CALCUL DE L'IMPORTANCE GINI")
    print("="*80)
    
    if target_col not in data.columns:
        raise ValueError(f"La colonne cible '{target_col}' est absente des données!")
    
    feature_columns = [col for col in data.columns if col != target_col]
    X = data[feature_columns].copy()
    y = data[target_col].copy()
    
    if sample_size is not None and len(data) > sample_size:
        print(f"  Échantillonnage : utilisation de {sample_size} lignes sur {len(data)}")
        sample_indices = np.random.choice(len(data), size=sample_size, replace=False)
        X = X.iloc[sample_indices]
        y = y.iloc[sample_indices]
    
    print(f"  Nombre de caractéristiques : {len(feature_columns)}")
    print(f"  Nombre d'échantillons : {len(X)}")
    print(f"  Méthode : {method}")
    print()
    
    # Entraîner le modèle selon la méthode choisie
    if method == 'xgboost':
        print("  Entraînement d'un modèle XGBoost pour calculer l'importance Gini...")
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        model.fit(X, y, verbose=False)
        
        importances = model.feature_importances_
        
    elif method == 'decision_tree':
        print("  Entraînement d'un arbre de décision pour calculer l'importance Gini...")
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
            criterion='gini'
        )
        model.fit(X, y)
        importances = model.feature_importances_
    
    else:
        raise ValueError(f"Méthode non reconnue : {method}. Utilisez 'xgboost' ou 'decision_tree'")
    
    # Créer un DataFrame avec les importances
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'gini_importance': importances
    }).sort_values('gini_importance', ascending=False)
    
    print(f"  Calcul terminé!")
    print()
    print("  Top 10 caractéristiques par importance Gini :")
    print(importance_df.head(10).to_string(index=False))
    print()
    
    return importance_df, model


def select_features_by_gini(data, target_col='DFlag', method='xgboost',
                            threshold=None, top_k=None, 
                            n_estimators=100, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, sample_size=None):
    """
    Sélectionne les caractéristiques basées sur l'importance Gini.
    
    Args:
        data: DataFrame avec les caractéristiques et la variable cible
        target_col: Nom de la colonne cible (par défaut 'DFlag')
        method: Méthode à utiliser ('xgboost' ou 'decision_tree')
        threshold: Seuil d'importance minimum (entre 0 et 1). Si None, utilise top_k
        top_k: Nombre de caractéristiques à conserver (utilisé si threshold est None)
        n_estimators: Nombre d'arbres pour XGBoost
        max_depth: Profondeur maximale des arbres
        learning_rate: Taux d'apprentissage
        subsample: Proportion d'échantillons utilisés
        colsample_bytree: Proportion de caractéristiques utilisées
        sample_size: Nombre d'échantillons à utiliser pour le calcul (None = toutes les données)
    
    Returns:
        Tuple (DataFrame avec seulement les caractéristiques sélectionnées + la variable cible, 
               DataFrame d'importance des caractéristiques)
    """
    print("="*80)
    print("SÉLECTION DE CARACTÉRISTIQUES PAR GINI")
    print("="*80)
    
    if target_col not in data.columns:
        raise ValueError(f"La colonne cible '{target_col}' est absente des données!")
    
    # Calculer l'importance Gini
    importance_df, model = calculate_gini_importance(
        data, target_col=target_col, method=method,
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, sample_size=sample_size
    )
    
    # Sélectionner les caractéristiques selon le critère
    if threshold is not None:
        selected_features = importance_df[importance_df['gini_importance'] >= threshold]['feature'].tolist()
        print(f"  Sélection par seuil (>= {threshold}) : {len(selected_features)} caractéristiques sélectionnées")
    elif top_k is not None:
        selected_features = importance_df.head(top_k)['feature'].tolist()
        print(f"  Sélection des top {top_k} : {len(selected_features)} caractéristiques sélectionnées")
    else:
        selected_features = importance_df[importance_df['gini_importance'] > 0]['feature'].tolist()
        print(f"  Sélection par défaut (importance > 0) : {len(selected_features)} caractéristiques sélectionnées")
    
    print()
    print("  Caractéristiques sélectionnées :")
    for i, feat in enumerate(selected_features[:20], 1):
        importance = importance_df[importance_df['feature'] == feat]['gini_importance'].values[0]
        print(f"    {i}. {feat} (importance: {importance:.6f})")
    if len(selected_features) > 20:
        print(f"    ... et {len(selected_features) - 20} autres")
    print()
    
    # Créer le DataFrame avec seulement les caractéristiques sélectionnées
    selected_data = data[[target_col] + selected_features].copy()
    
    print(f"  Forme initiale : {data.shape}")
    print(f"  Forme après sélection : {selected_data.shape}")
    print(f"  Réduction : {data.shape[1] - selected_data.shape[1]} caractéristiques supprimées")
    print()
    
    return selected_data, importance_df


def apply_gini_selection_to_files(project_path, window="FM12", segments=["green", "red"],
                                  splits=["train", "OOS", "OOT", "OOU"],
                                  method='xgboost', threshold=None, top_k=None,
                                  n_estimators=100, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, sample_size=50000):
    """
    Applique le prétraitement et la sélection Gini à tous les fichiers de données brutes.
    Les données prétraitées avec sélection Gini sont sauvegardées dans data/processed/.
    
    Args:
        project_path: Chemin du projet
        window: Fenêtre temporelle (ex: "FM12")
        segments: Liste des segments (ex: ["green", "red"])
        splits: Liste des splits (ex: ["train", "OOS", "OOT", "OOU"])
        method: Méthode à utiliser ('xgboost' ou 'decision_tree')
        threshold: Seuil d'importance minimum
        top_k: Nombre de caractéristiques à conserver
        n_estimators: Nombre d'arbres pour XGBoost
        max_depth: Profondeur maximale des arbres
        learning_rate: Taux d'apprentissage
        subsample: Proportion d'échantillons utilisés
        colsample_bytree: Proportion de caractéristiques utilisées
        sample_size: Nombre d'échantillons à utiliser pour calculer l'importance (sur les données d'entraînement)
    """
    print("="*80)
    print("PRÉTRAITEMENT ET SÉLECTION GINI")
    print("="*80)
    
    # 1. Charger et prétraiter les données d'entraînement brutes pour calculer l'importance
    print("\n1. Chargement et prétraitement des données d'entraînement brutes...")
    print(f"  Source : data/raw/{window}/[green,red]/train_{window[2:]}.csv")
    
    # Charger les données d'entraînement brutes
    train_chunks = []
    chunk_size = 50000
    
    for segment in segments:
        filename = f"train_{window[2:]}.csv"
        raw_path = os.path.join(project_path, "data", "raw", window, segment, filename)
        
        if os.path.exists(raw_path):
            print(f"  Lecture de {raw_path}...")
            for chunk in pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False):
                chunk_processed = preprocess_xgboost(chunk, target_col='DFlag')
                train_chunks.append(chunk_processed)
        else:
            print(f"  Fichier introuvable : {raw_path}")
    
    if not train_chunks:
        raise ValueError(f"Aucune donnée d'entraînement brute trouvée pour {window}!")
    
    print(f"  Concaténation de {len(train_chunks)} chunks...")
    train_data = pd.concat(train_chunks, ignore_index=True)
    print(f"  Données d'entraînement prétraitées : {train_data.shape}")
    print()
    
    print("2. Calcul de l'importance Gini sur les données d'entraînement prétraitées...")
    
    selected_train, importance_df = select_features_by_gini(
        train_data, target_col='DFlag', method=method,
        threshold=threshold, top_k=top_k,
        n_estimators=n_estimators, max_depth=max_depth, 
        learning_rate=learning_rate, subsample=subsample, 
        colsample_bytree=colsample_bytree, sample_size=sample_size
    )
    
    selected_features = [col for col in selected_train.columns if col != 'DFlag']
    
    print("\n3. Prétraitement et sélection Gini sur tous les fichiers bruts...")
    
    for segment in segments:
        for split in splits:
            if split == "OOU":
                raw_filename = f"{split}.sas7bdat"
            else:
                raw_filename = f"{split}_{window[2:]}.csv"
            
            raw_path = os.path.join(project_path, "data", "raw", window, segment, raw_filename)
            output_dir = os.path.join(project_path, "data", "processed", window, segment)
            os.makedirs(output_dir, exist_ok=True)
            
            output_filename = f"{split}_{window[2:]}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            if os.path.exists(raw_path):
                print(f"\n  Traitement : {raw_path}")
                
                try:
                    chunk_size = 50000
                    first_chunk = True
                    total_rows = 0
                    
                    # Lire le fichier brut (SAS ou CSV)
                    if split == "OOU":
                        print(f"    Lecture du fichier SAS OOU par chunks...")
                        try:
                            chunk_generator = read_sas_file(raw_path, chunk_size=chunk_size)
                            
                            chunk_num = 0
                            for chunk_df in chunk_generator:
                                chunk_processed = preprocess_xgboost(chunk_df, target_col='DFlag')
                                chunk_selected = chunk_processed[['DFlag'] + selected_features].copy()
                                
                                if first_chunk:
                                    chunk_selected.to_csv(output_path, index=False, mode='w', header=True)
                                    first_chunk = False
                                else:
                                    chunk_selected.to_csv(output_path, index=False, mode='a', header=False)
                                
                                total_rows += len(chunk_selected)
                                chunk_num += 1
                                if chunk_num % 10 == 0:
                                    print(f"      Chunks traités : {chunk_num} ({total_rows:,} lignes)...")
                            
                            print(f"      Total : {chunk_num} chunks, {total_rows:,} lignes")
                                        
                        except MemoryError:
                            print(f"    Erreur mémoire lors de la lecture du fichier SAS")
                            print(f"    Le fichier est peut-être trop volumineux pour la mémoire disponible")
                            print(f"    Suggestion : Réduire chunk_size ou augmenter la RAM")
                            raise
                        except Exception as e:
                            print(f"    Erreur lors de la lecture du fichier SAS : {str(e)}")
                            import traceback
                            traceback.print_exc()
                            raise
                    else:
                        for chunk in pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False):
                            chunk_processed = preprocess_xgboost(chunk, target_col='DFlag')
                            chunk_selected = chunk_processed[['DFlag'] + selected_features].copy()
                            
                            if first_chunk:
                                chunk_selected.to_csv(output_path, index=False, mode='w', header=True)
                                first_chunk = False
                            else:
                                chunk_selected.to_csv(output_path, index=False, mode='a', header=False)
                            
                            total_rows += len(chunk_selected)
                    
                    print(f"    Sauvegardé : {output_path} ({total_rows} lignes)")
                    
                except Exception as e:
                    print(f"    Erreur : {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  Fichier introuvable : {raw_path}")
    
    # Sauvegarder aussi le DataFrame d'importance
    importance_path = os.path.join(project_path, "data", "processed", window, "gini_importance.csv")
    os.makedirs(os.path.dirname(importance_path), exist_ok=True)
    importance_df.to_csv(importance_path, index=False)
    print(f"\n  Importance Gini sauvegardée : {importance_path}")
    
    print("\n" + "="*80)
    print("SÉLECTION GINI TERMINÉE")
    print("="*80)


if __name__ == "__main__":
    # Exemple d'utilisation
    # Sélectionner les top 20 caractéristiques basées sur l'importance Gini
    apply_gini_selection_to_files(
        project_path=PROJECT_PATH,
        window="FM12",
        segments=["green", "red"],
        splits=["train", "OOS", "OOT", "OOU"],
        method='xgboost',
        top_k=20,  # Garder les 20 caractéristiques les plus importantes
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        sample_size=50000  # Utiliser 50000 échantillons pour calculer l'importance
    )

