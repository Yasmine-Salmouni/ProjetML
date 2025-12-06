"""
Script de prétraitement des données pour XGBoost
"""

import os
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def read_sas_file(file_path, chunk_size=50000):
    """
    Lit un fichier SAS (.sas7bdat) par chunks en utilisant uniquement sas7bdat.
    Optimisé pour les gros fichiers - lit toujours par chunks pour éviter les problèmes de mémoire.
    
    Args:
        file_path: Chemin vers le fichier SAS
        chunk_size: Nombre de lignes par chunk (par défaut 50000)
    
    Returns:
        Générateur de DataFrames pandas (un chunk à la fois)
    
    Raises:
        Exception: Si sas7bdat n'est pas installé ou si la lecture échoue
    """
    # Vérifier que le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier SAS introuvable : {file_path}")
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  Taille du fichier : {file_size_mb:.2f} MB")
    print(f"  Lecture par chunks de {chunk_size:,} lignes avec sas7bdat...")
    
    try:
        from sas7bdat import SAS7BDAT
        
        def chunk_generator():
            with SAS7BDAT(file_path) as reader:
                chunk = []
                chunk_count = 0
                total_rows = 0
                
                for row in reader:
                    chunk.append(row)
                    total_rows += 1
                    
                    if len(chunk) >= chunk_size:
                        chunk_count += 1
                        df_chunk = pd.DataFrame(chunk)
                        if chunk_count % 10 == 0:
                            print(f"      Chunks lus : {chunk_count} ({total_rows:,} lignes)...")
                        yield df_chunk
                        chunk = []
                
                if chunk:
                    chunk_count += 1
                    df_chunk = pd.DataFrame(chunk)
                    print(f"      Dernier chunk : {chunk_count} ({total_rows:,} lignes au total)")
                    yield df_chunk
        
        print(f"  Fichier SAS prêt pour lecture par chunks")
        return chunk_generator()
            
    except ImportError:
        error_msg = (
            f"\nsas7bdat n'est pas installé !\n"
            f"   Installer avec : pip install sas7bdat\n"
            f"   sas7bdat est OBLIGATOIRE pour lire les fichiers SAS\n"
        )
        raise ImportError(error_msg)
    except MemoryError as e:
        error_msg = (
            f"\nErreur mémoire lors de la lecture du fichier SAS : {file_path}\n"
            f"   Taille du fichier : {file_size_mb:.2f} MB\n"
            f"   Le fichier est peut-être trop volumineux pour la mémoire disponible\n"
            f"\nSolutions possibles :\n"
            f"  1. Réduire chunk_size (actuellement {chunk_size:,})\n"
            f"  2. Augmenter la RAM disponible\n"
            f"  3. Convertir le fichier SAS en CSV avec un autre outil (SAS, R, etc.)\n"
        )
        raise MemoryError(error_msg)
    except Exception as e:
        error_msg = (
            f"\nErreur lors de la lecture du fichier SAS : {file_path}\n"
            f"   Taille du fichier : {file_size_mb:.2f} MB\n"
            f"   Erreur : {str(e)}\n"
            f"   Type : {type(e).__name__}\n"
            f"\nSolutions possibles :\n"
            f"  1. Vérifier que le fichier n'est pas corrompu\n"
            f"  2. Vérifier que sas7bdat est correctement installé : pip install sas7bdat\n"
            f"  3. Convertir le fichier SAS en CSV avec un autre outil (SAS, R, etc.)\n"
        )
        import traceback
        print(f"\nDétails de l'erreur :")
        traceback.print_exc()
        raise Exception(error_msg)

#Gestion des dates
'''
'''
def yyqq_to_date(yyqq):
    """
    Convertit un format YYQQ en date.
    """
    try:
        if pd.isna(yyqq) or yyqq == '':
            return pd.NaT
        
        yyqq_str = str(yyqq)
        if len(yyqq_str) < 5:
            return pd.NaT
        
        # Si ce ne sont pas des chiffres, ValueError sera levé
        yy = int(yyqq_str[1:3])
        qq = yyqq_str[3:5]
        
        year = 2000 + yy if yy < 50 else 1900 + yy
        month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}.get(qq, 1)
        
        return pd.Timestamp(year=year, month=month, day=1)
    
    except (ValueError, IndexError, KeyError):
        # En cas d'erreur (format invalide, index hors limites, etc.), retourner NaT
        return pd.NaT


def extract_date_features(data, date_col='Origination_date'):
    """
    Extrait des features numériques à partir d'une colonne de date
    """
    if date_col not in data.columns:
        return data
    
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    
    # Extraire des features numériques
    data[f'{date_col}_year'] = data[date_col].dt.year
    data[f'{date_col}_month'] = data[date_col].dt.month
    data[f'{date_col}_quarter'] = data[date_col].dt.quarter
    
    # Supprimer la colonne de date originale
    data = data.drop(columns=[date_col])
    
    return data


def remove_id_columns(data):
    """
    Supprime la colonne 'Loanref'.
    
    Args:
        data: DataFrame à nettoyer    
    Returns:
        DataFrame nettoyé
    """
    data = data.copy()
    
    if 'Loanref' in data.columns:
        unique_ratio = data['Loanref'].nunique() / len(data) if len(data) > 0 else 0
        print(f"  Loanref supprimée ({unique_ratio*100:.1f}% de valeurs uniques)")
        data = data.drop(columns=['Loanref'])
    else:
        print("  Colonne Loanref non trouvée (déjà supprimée ou absente)")
    
    return data


def encode_categorical_variables(data, target_col='DFlag'):
    """
    Encode les variables catégorielles en utilisant Label Encoding
    """
    data = data.copy()
    label_encoders = {}
    
    categorical_cols = data.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = data[col].fillna('__MISSING__')
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
            print(f"  Encodage Label Encoding : {col}")
    
    return data, label_encoders


def check_business_logic(data):
    """
    Vérifie la logique métier et corrige les valeurs aberrantes
    """
    data = data.copy()
    
    # Colonnes qui ne devraient pas être négatives
    non_negative_cols = [
        'Credit_Score', 'Number_of_units', 'CLoan_to_value', 
        'Debt_to_income', 'OLoan_to_value'
    ]
    
    for col in non_negative_cols:
        if col in data.columns:
            negative_count = (data[col] < 0).sum()
            if negative_count > 0:
                print(f"  {col} : {negative_count} valeurs négatives remplacées par NaN")
                data.loc[data[col] < 0, col] = np.nan
    
    return data


def detect_data_leakage(data, target_col='DFlag'):
    """
    Détecte et supprime les colonnes qui causent du data leakage.
    
    Args:
        data: DataFrame à vérifier (doit contenir DFlag)
        target_col: Nom de la colonne cible (par défaut 'DFlag')
    
    Returns:
        DataFrame nettoyé (avec DFlag conservée)
    """
    data = data.copy()
    leakage_cols = []
    
    if target_col not in data.columns:
        print(f"  Attention : La colonne cible '{target_col}' n'est pas présente dans les données")
        return data
    
    if 'Loanref' in data.columns:
        leakage_cols.append('Loanref')
        print(f"  Loanref détecté (devrait avoir été supprimé précédemment) - suppression")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) > 0 and len(data) > 0:
        try:
            correlations = data[numeric_cols + [target_col]].corr()[target_col]
            
            for col in numeric_cols:
                if col in correlations.index:
                    corr_value = abs(correlations[col])
                    if corr_value > 0.99:
                        leakage_cols.append(col)
                        print(f"  Colonne suspecte (corrélation {corr_value*100:.2f}% avec {target_col}) : {col}")
        except Exception as e:
            print(f"  Attention : Impossible de calculer les corrélations : {str(e)}")
    
    cols_to_remove = [col for col in leakage_cols if col != target_col]
    
    if cols_to_remove:
        data = data.drop(columns=cols_to_remove)
        print(f"  Total de colonnes supprimées pour data leakage : {len(cols_to_remove)}")
    else:
        print("  Aucune colonne suspecte de data leakage détectée")
    
    return data


def convert_to_numeric(data, target_col='DFlag'):
    """
    Convertit toutes les colonnes numériques en float64
    (sauf la colonne cible si elle existe)
    """
    data = data.copy()
    
    for col in data.columns:
        if col == target_col:
            continue
        
        # Si la colonne est déjà numérique, la convertir en float64
        if data[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
        # Si c'est booléen, convertir en int
        elif data[col].dtype == 'bool':
            data[col] = data[col].astype('int64')
    
    return data


def preprocess_xgboost(data, target_col='DFlag'):
    """
    Fonction principale de prétraitement pour XGBoost
    
    Args:
        data: DataFrame pandas (doit contenir la colonne 'DFlag' comme variable cible)
        target_col: Nom de la colonne cible (par défaut 'DFlag')
    
    Returns:
        DataFrame prétraité (avec DFlag conservée comme variable cible y)
    """
    print("Début du prétraitement pour XGBoost...")
    print(f"  Forme initiale : {data.shape}")
    
    data = data.copy()
    
    # 1. Gérer les dates AVANT de supprimer les IDs (car Loanref est utilisé pour créer la date)
    print("\n1. Traitement des dates...")
    if 'Loanref' in data.columns:
        print("  Extraction de la date depuis Loanref...")
        data['Origination_date'] = data['Loanref'].apply(yyqq_to_date)
        data = extract_date_features(data, date_col='Origination_date')
        print("  Features de date créées : Origination_date_year, Origination_date_month, Origination_date_quarter")
    
    print("\n2. Suppression des identifiants...")
    data = remove_id_columns(data)
    
    print("\n3. Détection du data leakage...")
    data = detect_data_leakage(data, target_col=target_col)
    
    date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    for date_col in date_cols:
        if date_col != target_col:
            data = extract_date_features(data, date_col=date_col)
    
    print("\n4. Encodage des variables catégorielles...")
    data, label_encoders = encode_categorical_variables(data, target_col=target_col)
    
    print("\n5. Vérification de la logique métier...")
    data = check_business_logic(data)
    
    print("\n6. Conversion en types numériques...")
    data = convert_to_numeric(data, target_col=target_col)
    
    if 'CLoan_to_value' in data.columns:
        # Si les valeurs sont > 1, elles sont probablement en pourcentage
        max_val = data['CLoan_to_value'].max()
        if not pd.isna(max_val) and max_val > 1:
            data['CLoan_to_value'] = data['CLoan_to_value'] / 100.0
            print("  CLoan_to_value : valeurs divisées par 100")
    
    if 'OLoan_to_value' in data.columns:
        max_val = data['OLoan_to_value'].max()
        if not pd.isna(max_val) and max_val > 1:
            data['OLoan_to_value'] = data['OLoan_to_value'] / 100.0
            print("  OLoan_to_value : valeurs divisées par 100")
    
    print(f"\nPrétraitement terminé. Forme finale : {data.shape}")
    print(f"  Colonnes : {list(data.columns)}")
    
    return data


def process_and_save_all(project_path, windows=["FM12"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"]):
    """
    Traite et sauvegarde tous les fichiers spécifiés
    
    Args:
        project_path: Chemin du projet
        windows: Liste des fenêtres (ex: ["FM12"])
        segments: Liste des segments (ex: ["green", "red"])
        splits: Liste des splits (ex: ["train", "OOS", "OOT", "OOU"])
    """
    for window in windows:
        for segment in segments:
            for split in splits:
                # Déterminer le nom du fichier
                if split == "OOU":
                    filename = f"{split}.sas7bdat"
                else:
                    filename = f"{split}_{window[2:]}.csv"
                
                raw_path = os.path.join(project_path, "data", "raw", window, segment, filename)
                
                if os.path.exists(raw_path):
                    print(f"\n{'='*80}")
                    print(f"Traitement : {raw_path}")
                    print(f"{'='*80}")
                    
                    try:
                        # Lire le fichier
                        if split == "OOU":
                            # Lire le fichier SAS par chunks (comme les fichiers CSV)
                            print(f"  Lecture du fichier SAS par chunks...")
                            
                            # Créer le dossier de destination
                            save_dir = os.path.join(project_path, "data", "processed", window, segment)
                            os.makedirs(save_dir, exist_ok=True)
                            
                            # Nom du fichier de sortie
                            save_filename = f"{split}_{window[2:]}.csv"
                            save_path = os.path.join(save_dir, save_filename)
                            
                            # Lire et traiter par chunks
                            chunk_size = 50000
                            chunk_generator = read_sas_file(raw_path, chunk_size=chunk_size)
                            
                            first_chunk = True
                            chunk_num = 1
                            total_rows = 0
                            
                            for chunk in chunk_generator:
                                print(f"  Traitement du chunk {chunk_num} ({len(chunk)} lignes)...")
                                
                                # Prétraiter le chunk
                                chunk_processed = preprocess_xgboost(chunk, target_col='DFlag')
                                
                                # Sauvegarder (avec en-tête seulement pour le premier chunk)
                                if first_chunk:
                                    chunk_processed.to_csv(save_path, index=False, mode='w', header=True)
                                    first_chunk = False
                                    print(f"  Chunk {chunk_num} traité et sauvegardé avec en-tête")
                                else:
                                    chunk_processed.to_csv(save_path, index=False, mode='a', header=False)
                                    print(f"  Chunk {chunk_num} traité et ajouté au fichier")
                                
                                total_rows += len(chunk_processed)
                                chunk_num += 1
                            
                            print(f"\n  Sauvegardé : {save_path} ({total_rows:,} lignes au total)")
                        else:
                            # Lire par chunks pour gérer les gros fichiers
                            print("  Lecture par chunks (fichier volumineux)...")
                            chunk_size = 50000  # Nombre de lignes par chunk
                            
                            # Créer le dossier de destination
                            save_dir = os.path.join(project_path, "data", "processed", window, segment)
                            os.makedirs(save_dir, exist_ok=True)
                            
                            # Nom du fichier de sortie
                            save_filename = f"{split}_{window[2:]}.csv"
                            save_path = os.path.join(save_dir, save_filename)
                            
                            # Lire et traiter par chunks
                            first_chunk = True
                            chunk_num = 1
                            total_rows = 0
                            
                            for chunk in pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False):
                                if len(chunk) == 0:
                                    break
                                
                                print(f"  Traitement du chunk {chunk_num} ({len(chunk)} lignes)...")
                                
                                # Prétraiter le chunk
                                chunk_processed = preprocess_xgboost(chunk, target_col='DFlag')
                                
                                # Sauvegarder (avec en-tête seulement pour le premier chunk)
                                if first_chunk:
                                    chunk_processed.to_csv(save_path, index=False, mode='w', header=True)
                                    first_chunk = False
                                    print(f"  Chunk {chunk_num} traité et sauvegardé avec en-tête")
                                else:
                                    chunk_processed.to_csv(save_path, index=False, mode='a', header=False)
                                    print(f"  Chunk {chunk_num} traité et ajouté au fichier")
                                
                                total_rows += len(chunk_processed)
                                chunk_num += 1
                            
                            print(f"\n Sauvegardé : {save_path} ({total_rows} lignes au total)")
                        
                    except Exception as e:
                        print(f"\n Erreur lors du traitement de {raw_path}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"  Fichier introuvable : {raw_path}")


def load_processed_data(project_path, windows=["FM12"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"]):
    """
    Charge les données prétraitées (gère les gros fichiers par chunks)
    
    Args:
        project_path: Chemin du projet
        windows: Liste des fenêtres
        segments: Liste des segments
        splits: Liste des splits
    
    Returns:
        DataFrame concaténé
    """
    dataframes = []
    for window in windows:
        for segment in segments:
            for split in splits:
                filename = f"{split}_{window[2:]}.csv"
                file_path = os.path.join(project_path, "data", "processed", window, segment, filename)
                if os.path.exists(file_path):
                    try:
                        # Essayer de lire normalement d'abord (petits fichiers)
                        df = pd.read_csv(file_path, low_memory=False, engine='c')
                        dataframes.append(df)
                        print(f"Chargé : {file_path} ({df.shape[0]} lignes)")
                    except (pd.errors.ParserError, MemoryError, Exception) as e:
                        # Si erreur, lire par chunks avec engine python (plus lent mais moins de mémoire)
                        print(f"  Fichier volumineux détecté, lecture par chunks (engine python) : {file_path}")
                        chunk_list = []
                        chunk_size = 10000  # Chunks plus petits pour éviter les problèmes de mémoire
                        
                        try:
                            chunk_count = 0
                            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, engine='python'):
                                chunk_list.append(chunk)
                                chunk_count += 1
                                if chunk_count % 10 == 0:
                                    print(f"    Chunks lus : {chunk_count}...")
                            
                            if chunk_list:
                                print(f"    Concaténation de {len(chunk_list)} chunks...")
                                df = pd.concat(chunk_list, ignore_index=True)
                                dataframes.append(df)
                                print(f"  Chargé : {file_path} ({df.shape[0]} lignes)")
                            else:
                                print(f"  Fichier vide : {file_path}")
                        except Exception as e2:
                            print(f"  Erreur lors de la lecture par chunks : {str(e2)}")
                            print(f"     Le fichier est peut-être corrompu ou trop volumineux pour la mémoire disponible")
                            raise
                else:
                    print(f"  Fichier introuvable : {file_path}")
    
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()
