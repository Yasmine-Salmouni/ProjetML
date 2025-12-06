"""
Script pour entraîner un modèle XGBoost sur les données prétraitées
python models/train_xgboost.py
"""

import os
import sys
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from preprocess_xgboost import load_processed_data

def train_xgboost_model(project_path, window="FM12", segments=["green", "red"], use_gini_selection=False):
    """
    Entraîne un modèle XGBoost sur les données d'entraînement.
    
    Args:
        project_path: Chemin du projet
        window: Fenêtre temporelle (ex: "FM12")
        segments: Liste des segments à fusionner (ex: ["green", "red"])
        use_gini_selection: Si True, utilise les données avec sélection Gini (data/processed_gini/)
                          Si False, utilise les données complètes (data/processed/)
    
    Returns:
        Modèle XGBoost entraîné
    """
    print("="*80)
    print("ENTRAÎNEMENT DU MODÈLE XGBOOST")
    print("="*80)
    print(f"Chemin du projet : {project_path}")
    print(f"Fenêtre : {window}")
    print(f"Segments : {segments}")
    print(f"Sélection Gini : {'Activée' if use_gini_selection else 'Désactivée'}")
    print()
    
    data_dir = "processed_gini" if use_gini_selection else "processed"
    print("1. Chargement des données d'entraînement prétraitées par chunks...")
    print(f"  Source : data/{data_dir}/{window}/[green,red]/train_{window[2:]}.csv")
    
    import os
    files_to_load = []
    for segment in segments:
        filename = f"train_{window[2:]}.csv"
        file_path = os.path.join(project_path, "data", data_dir, window, segment, filename)
        if os.path.exists(file_path):
            files_to_load.append(file_path)
    
    if not files_to_load:
        if use_gini_selection:
            raise ValueError(
                f"Aucune donnée d'entraînement trouvée dans data/{data_dir}/{window}/!\n"
                f"Assurez-vous d'avoir exécuté le prétraitement avec sélection Gini (run_preprocessing.py)"
            )
        else:
            raise ValueError(
                f"Aucune donnée d'entraînement trouvée dans data/{data_dir}/{window}/!\n"
                f"Assurez-vous d'avoir exécuté le prétraitement avec run_preprocessing.py"
            )
    
    print("  Lecture du premier chunk pour identifier la structure...")
    chunk_size = 10000
    first_chunk = pd.read_csv(files_to_load[0], nrows=chunk_size, engine='python')
    
    if 'DFlag' not in first_chunk.columns:
        raise ValueError("La colonne 'DFlag' (variable cible) est absente des données!")
    
    feature_columns = [col for col in first_chunk.columns if col != 'DFlag']
    print(f"  Features identifiées : {len(feature_columns)} colonnes")
    print()
    
    print("2. Création du modèle XGBoost...")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    print("  Paramètres du modèle :")
    print(f"    - Objectif : binary:logistic")
    print(f"    - Nombre d'arbres : {model.n_estimators}")
    print(f"    - Profondeur max : {model.max_depth}")
    print(f"    - Taux d'apprentissage : {model.learning_rate}")
    print()
    
    print("3. Accumulation des chunks et entraînement du modèle...")
    print()
    
    total_rows = 0
    total_chunks = 0
    accumulated_chunks = []
    max_chunks_in_memory = 5
    
    for file_path in files_to_load:
        print(f"  Traitement du fichier : {os.path.basename(file_path)}")
        chunk_count = 0
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, engine='python'):
            accumulated_chunks.append(chunk)
            total_rows += len(chunk)
            total_chunks += 1
            chunk_count += 1
            
            if len(accumulated_chunks) >= max_chunks_in_memory:
                batch_df = pd.concat(accumulated_chunks, ignore_index=True)
                X_batch = batch_df[feature_columns]
                y_batch = batch_df['DFlag']
                
                if total_chunks == len(accumulated_chunks):
                    model.fit(X_batch, y_batch, verbose=False)
                else:
                    model.fit(
                        X_batch, 
                        y_batch,
                        xgb_model=model.get_booster(),
                        verbose=False
                    )
                
                accumulated_chunks = []
                
                if chunk_count % 10 == 0:
                    print(f"    Chunks traités : {chunk_count} ({total_rows} lignes au total)...")
        
        print(f"  Fichier terminé : {chunk_count} chunks, {total_rows} lignes")
        print()
    
    if accumulated_chunks:
        batch_df = pd.concat(accumulated_chunks, ignore_index=True)
        X_batch = batch_df[feature_columns]
        y_batch = batch_df['DFlag']
        model.fit(
            X_batch, 
            y_batch,
            xgb_model=model.get_booster(),
            verbose=False
        )
    
    print(f"  Entraînement terminé : {total_chunks} chunks, {total_rows} lignes au total")
    print()
    
    print("4. Analyse de la distribution des classes...")
    sample_size = min(100000, total_rows)
    sample_chunks = []
    for file_path in files_to_load:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, engine='python'):
            sample_chunks.append(chunk)
            if sum(len(c) for c in sample_chunks) >= sample_size:
                break
        if sum(len(c) for c in sample_chunks) >= sample_size:
            break
    
    sample_df = pd.concat(sample_chunks, ignore_index=True)
    y_sample = sample_df['DFlag']
    print(f"  Échantillon analysé : {len(y_sample)} lignes")
    print(f"  Distribution de DFlag (estimée) :")
    print(f"    - 0 (pas de défaut) : {(y_sample == 0).sum()} ({((y_sample == 0).sum() / len(y_sample) * 100):.2f}%)")
    print(f"    - 1 (défaut) : {(y_sample == 1).sum()} ({((y_sample == 1).sum() / len(y_sample) * 100):.2f}%)")
    print()
    
    print("  Modèle entraîné avec succès!")
    print()
    
    print("5. Évaluation du modèle sur un échantillon des données d'entraînement...")
    
    X_sample = sample_df[feature_columns]
    y_sample = sample_df['DFlag']
    
    y_pred_sample = model.predict(X_sample)
    accuracy_sample = accuracy_score(y_sample, y_pred_sample)
    
    print(f"  Précision (accuracy) sur échantillon : {accuracy_sample:.4f} ({accuracy_sample*100:.2f}%)")
    print(f"  (Échantillon de {len(y_sample)} lignes sur {total_rows} total)")
    print()
    print("  Rapport de classification (échantillon) :")
    print(classification_report(y_sample, y_pred_sample, target_names=['Pas de défaut', 'Défaut']))
    print()
    print("  Matrice de confusion (échantillon) :")
    print(confusion_matrix(y_sample, y_pred_sample))
    print()
    
    print("6. Sauvegarde du modèle...")
    model_dir = os.path.join(project_path, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_filename = f"xgboost_model_{window}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  Modèle sauvegardé : {model_path}")
    print()
    
    print("7. Importance des features (top 10) :")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    print()
    
    print("="*80)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*80)
    
    return model

if __name__ == "__main__":
    model = train_xgboost_model(
        project_path=PROJECT_PATH,
        window="FM12",
        segments=["green", "red"]
    )
