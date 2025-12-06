"""
Script pour tester le modèle XGBoost sur les données OOS (Out-of-Sample)
Permet de détecter le surapprentissage en comparant les performances
sur les données d'entraînement vs les données OOS
"""
#python models/test_xgboost.py --test-split OOS (pae défaut)
#python models/test_xgboost.py --test-split OOT (OOT)


import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle
import matplotlib.pyplot as plt

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

def test_xgboost_model(project_path, window="FM12", segments=["green", "red"], test_split="OOS"):
    """
    Teste le modèle XGBoost sur les données de test (OOS, OOT, ou OOU).
    
    Args:
        project_path: Chemin du projet
        window: Fenêtre temporelle (ex: "FM12")
        segments: Liste des segments à fusionner (ex: ["green", "red"])
        test_split: Type de données de test ("OOS", "OOT", ou "OOU")
    
    Returns:
        Dictionnaire avec les métriques de performance
    """
    print("="*80)
    print(f"TEST DU MODÈLE XGBOOST SUR LES DONNÉES {test_split}")
    print("="*80)
    print(f"Chemin du projet : {project_path}")
    print(f"Fenêtre : {window}")
    print(f"Segments : {segments}")
    print(f"Split de test : {test_split}")
    print()
    
    # 1. Charger le modèle entraîné
    print("1. Chargement du modèle entraîné...")
    model_dir = os.path.join(project_path, "models")
    model_filename = f"xgboost_model_{window}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle non trouvé : {model_path}\n"
            f"Assurez-vous d'avoir entraîné le modèle avec models/train_xgboost.py"
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"  Modèle chargé : {model_path}")
    print()
    
    # 2. Charger les données de test prétraitées par chunks
    print(f"2. Chargement des données de test {test_split} prétraitées par chunks...")
    print(f"  Source : data/processed/{window}/[green,red]/{test_split}_{window[2:]}.csv")
    
    # Trouver les fichiers à charger
    files_to_load = []
    for segment in segments:
        filename = f"{test_split}_{window[2:]}.csv"
        file_path = os.path.join(project_path, "data", "processed", window, segment, filename)
        if os.path.exists(file_path):
            files_to_load.append(file_path)
    
    if not files_to_load:
        raise ValueError(
            f"Aucune donnée de test {test_split} trouvée dans data/processed/{window}/!\n"
            f"Assurez-vous d'avoir exécuté le prétraitement avec run_preprocessing.py"
        )
    
    # Lire le premier chunk pour obtenir les colonnes
    print("  Lecture du premier chunk pour identifier la structure...")
    chunk_size = 10000
    first_chunk = pd.read_csv(files_to_load[0], nrows=chunk_size, engine='python')
    
    if 'DFlag' not in first_chunk.columns:
        raise ValueError("La colonne 'DFlag' (variable cible) est absente des données de test!")
    
    feature_columns = [col for col in first_chunk.columns if col != 'DFlag']
    print(f"  Features identifiées : {len(feature_columns)} colonnes")
    print()
    
    # 3. Faire des prédictions par chunks (SANS montrer y_test au modèle)
    print("3. Prédictions du modèle par chunks (sans montrer les réponses)...")
    print("  Le modèle ne voit que X_test, pas y_test")
    print()
    
    # Stocker toutes les prédictions et les vraies valeurs
    all_y_test = []
    all_y_pred = []
    all_y_pred_proba = []
    total_rows = 0
    
    for file_path in files_to_load:
        print(f"  Traitement du fichier : {os.path.basename(file_path)}")
        chunk_count = 0
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, engine='python'):
            # Séparer X et y
            X_chunk = chunk[feature_columns]
            y_chunk = chunk['DFlag']  # La vraie réponse (on ne la montre PAS au modèle)
            
            # Faire des prédictions sur ce chunk
            y_pred_chunk = model.predict(X_chunk)
            y_pred_proba_chunk = model.predict_proba(X_chunk)[:, 1]
            
            # Stocker les résultats
            all_y_test.extend(y_chunk.values)
            all_y_pred.extend(y_pred_chunk)
            all_y_pred_proba.extend(y_pred_proba_chunk)
            
            total_rows += len(chunk)
            chunk_count += 1
            
            if chunk_count % 10 == 0:
                print(f"    Chunks traités : {chunk_count} ({total_rows} lignes au total)...")
        
        print(f"  Fichier terminé : {chunk_count} chunks, {total_rows} lignes")
        print()
    
    # Convertir en arrays numpy pour les calculs
    y_test = np.array(all_y_test)
    y_pred = np.array(all_y_pred)
    y_pred_proba = np.array(all_y_pred_proba)
    
    print(f"  Prédictions effectuées : {len(y_pred)} prédictions au total")
    print()
    
    # 4. Afficher la distribution de DFlag
    print("4. Distribution de DFlag dans les données de test :")
    print(f"    - 0 (pas de défaut) : {(y_test == 0).sum()} ({((y_test == 0).sum() / len(y_test) * 100):.2f}%)")
    print(f"    - 1 (défaut) : {(y_test == 1).sum()} ({((y_test == 1).sum() / len(y_test) * 100):.2f}%)")
    print()
    
    # 5. Comparer les prédictions avec la réalité
    print("5. Comparaison des prédictions avec la réalité...")
    print(f"  Total de prédictions : {len(y_pred)}")
    
    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculer AUC-ROC si possible
    try:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc_roc = None
        print("   AUC-ROC non calculable (problème avec les classes)")
    
    print(f"  Précision (Accuracy) : {accuracy:.4f} ({accuracy*100:.2f}%)")
    if auc_roc is not None:
        print(f"  AUC-ROC : {auc_roc:.4f}")
    print()
    
    # 6. Rapport de classification détaillé
    print("6. Rapport de classification détaillé :")
    print(classification_report(y_test, y_pred, target_names=['Pas de défaut', 'Défaut']))
    print()
    
    # 7. Matrice de confusion
    print("7. Matrice de confusion :")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print()
    print("  Légende :")
    print("    [Vrais Négatifs]  [Faux Positifs]")
    print("    [Faux Négatifs]   [Vrais Positifs]")
    print()
    
    # 8. Sauvegarder les résultats
    print("8. Sauvegarde des résultats...")
    results_dir = os.path.join(project_path, "outputs", "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    results_filename = f"results_{test_split}_{window}.txt"
    results_path = os.path.join(results_dir, results_filename)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Résultats du test sur {test_split} - {window}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Précision (Accuracy) : {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        if auc_roc is not None:
            f.write(f"AUC-ROC : {auc_roc:.4f}\n")
        f.write("\nMatrice de confusion :\n")
        f.write(str(cm) + "\n")
        f.write("\nRapport de classification :\n")
        f.write(classification_report(y_test, y_pred, target_names=['Pas de défaut', 'Défaut']))
    
    print(f"  Résultats sauvegardés : {results_path}")
    print()
    
    print("="*80)
    print(f"TEST TERMINÉ")
    print("="*80)
    
    # Retourner les métriques
    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

if __name__ == "__main__":
    import argparse
    
    # Parser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description='Tester le modèle XGBoost sur les données de test')
    parser.add_argument(
        '--test-split',
        type=str,
        choices=['OOS', 'OOT', 'OOU'],
        default='OOS',
        help='Type de données de test à utiliser : OOS (Out-of-Sample), OOT (Out-of-Time), ou OOU (Out-of-Universe). Par défaut : OOS'
    )
    parser.add_argument(
        '--window',
        type=str,
        default='FM12',
        help='Fenêtre temporelle (ex: FM12). Par défaut : FM12'
    )
    
    args = parser.parse_args()
    
    # Tester sur le split spécifié
    results = test_xgboost_model(
        project_path=PROJECT_PATH,
        window=args.window,
        segments=["green", "red"],
        test_split=args.test_split
    )

