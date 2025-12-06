"""
Module pour la sélection de variables basée sur le coefficient de Gini
Peut être utilisé avec n'importe quel pipeline de prétraitement
"""

import os
import pandas as pd
import numpy as np

def calculate_gini_impurity(y):
    """
    Calcule l'impureté de Gini pour une variable cible.
    
    Args:
        y: Série pandas avec les valeurs de la variable cible
    
    Returns:
        Coefficient de Gini (entre 0 et 1, où 0 = pur, 1 = impur)
    """
    if len(y) == 0:
        return 0.0
    
    # Compter les occurrences de chaque classe
    value_counts = y.value_counts()
    proportions = value_counts / len(y)
    
    # Calculer l'impureté de Gini: 1 - sum(p_i^2)
    gini = 1 - (proportions ** 2).sum()
    
    return gini

def calculate_gini_gain(data, feature_col, target_col='DFlag'):
    """
    Calcule le gain de Gini pour une variable par rapport à la variable cible.
    Le gain de Gini mesure la réduction de l'impureté obtenue en utilisant cette variable.
    
    Args:
        data: DataFrame pandas
        feature_col: Nom de la colonne de la variable à évaluer
        target_col: Nom de la colonne cible (par défaut 'DFlag')
    
    Returns:
        Gain de Gini (plus élevé = meilleure variable)
    """
    if feature_col not in data.columns or target_col not in data.columns:
        return 0.0
    
    # Supprimer les valeurs manquantes pour cette variable
    data_clean = data[[feature_col, target_col]].dropna()
    
    if len(data_clean) == 0:
        return 0.0
    
    # Impureté de Gini de la variable cible (avant division)
    gini_parent = calculate_gini_impurity(data_clean[target_col])
    
    # Pour les variables continues, utiliser des seuils
    # Pour les variables discrètes, calculer directement
    if data_clean[feature_col].dtype in ['int64', 'int32', 'float64', 'float32']:
        # Variable numérique: essayer différents seuils
        unique_values = sorted(data_clean[feature_col].unique())
        
        if len(unique_values) <= 1:
            return 0.0
        
        best_gain = 0.0
        
        # Essayer des seuils (médiane et quartiles pour efficacité)
        thresholds = []
        if len(unique_values) > 10:
            # Utiliser des quantiles pour les grandes variables
            thresholds = [
                data_clean[feature_col].quantile(0.25),
                data_clean[feature_col].quantile(0.5),
                data_clean[feature_col].quantile(0.75)
            ]
        else:
            # Utiliser toutes les valeurs uniques pour les petites variables
            thresholds = unique_values[:-1]  # Exclure la dernière valeur
        
        for threshold in thresholds:
            # Diviser en deux groupes: <= seuil et > seuil
            left_mask = data_clean[feature_col] <= threshold
            right_mask = ~left_mask
            
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            
            # Calculer l'impureté de Gini pour chaque groupe
            gini_left = calculate_gini_impurity(data_clean[left_mask][target_col])
            gini_right = calculate_gini_impurity(data_clean[right_mask][target_col])
            
            # Poids de chaque groupe
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            n_total = len(data_clean)
            
            # Impureté de Gini pondérée
            weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
            
            # Gain de Gini
            gain = gini_parent - weighted_gini
            best_gain = max(best_gain, gain)
        
        return best_gain
    else:
        # Variable catégorielle: calculer directement
        gini_children = 0.0
        n_total = len(data_clean)
        
        for value in data_clean[feature_col].unique():
            mask = data_clean[feature_col] == value
            n_value = mask.sum()
            
            if n_value > 0:
                gini_value = calculate_gini_impurity(data_clean[mask][target_col])
                gini_children += (n_value / n_total) * gini_value
        
        # Gain de Gini
        gain = gini_parent - gini_children
        return max(0.0, gain)

def select_features_by_gini(data, target_col='DFlag', min_gini_gain=0.0, top_k=None, save_scores_path=None):
    """
    Sélectionne les variables basées sur leur gain de Gini.
    
    Args:
        data: DataFrame pandas
        target_col: Nom de la colonne cible (par défaut 'DFlag')
        min_gini_gain: Gain de Gini minimum requis (par défaut 0.0)
        top_k: Nombre de meilleures variables à garder (None = garder toutes celles qui passent le seuil)
        save_scores_path: Chemin pour sauvegarder les scores de Gini (None = ne pas sauvegarder)
    
    Returns:
        Tuple (DataFrame avec seulement les variables sélectionnées, DataFrame avec tous les scores de Gini)
    """
    if target_col not in data.columns:
        print(f"Attention: La colonne cible '{target_col}' n'est pas présente")
        return data, pd.DataFrame()
    
    print(f"\nSélection de variables basée sur le coefficient de Gini...")
    print(f"  Colonnes initiales: {len(data.columns)}")
    
    # Calculer le gain de Gini pour chaque variable
    feature_cols = [col for col in data.columns if col != target_col]
    gini_gains = {}
    
    print(f"  Calcul des scores de Gini pour {len(feature_cols)} variables...")
    for i, col in enumerate(feature_cols, 1):
        if i % 5 == 0 or i == len(feature_cols):
            print(f"    Progression: {i}/{len(feature_cols)} variables traitées...")
        gain = calculate_gini_gain(data, col, target_col)
        gini_gains[col] = gain
    
    # Créer un DataFrame avec les résultats
    gini_df = pd.DataFrame({
        'feature': list(gini_gains.keys()),
        'gini_gain': list(gini_gains.values())
    }).sort_values('gini_gain', ascending=False)
    
    print(f"\n  Top 10 des variables par gain de Gini:")
    print(gini_df.head(10).to_string(index=False))
    
    # Sauvegarder les scores si un chemin est fourni
    if save_scores_path is not None:
        os.makedirs(os.path.dirname(save_scores_path), exist_ok=True)
        gini_df.to_csv(save_scores_path, index=False)
        print(f"\n  Scores de Gini sauvegardés : {save_scores_path}")
    
    # Sélectionner les variables
    if top_k is not None:
        selected_features = gini_df.head(top_k)['feature'].tolist()
        print(f"\n  Sélection des {top_k} meilleures variables")
    else:
        selected_features = gini_df[gini_df['gini_gain'] >= min_gini_gain]['feature'].tolist()
        print(f"\n  Sélection des variables avec gain de Gini >= {min_gini_gain}")
    
    # Toujours inclure la variable cible
    selected_features.append(target_col)
    
    print(f"  Colonnes sélectionnées: {len(selected_features)}")
    
    return data[selected_features], gini_df

def load_gini_scores(project_path, window="FM12", segment="green"):
    """
    Charge les scores de Gini sauvegardés pour une fenêtre et un segment donnés.
    
    Args:
        project_path: Chemin du projet
        window: Fenêtre temporelle (ex: "FM12")
        segment: Segment (ex: "green" ou "red")
    
    Returns:
        DataFrame avec les scores de Gini, ou None si le fichier n'existe pas
    """
    gini_scores_path = os.path.join(project_path, "outputs", "gini_scores", window, segment, f"gini_scores_{window}.csv")
    
    if os.path.exists(gini_scores_path):
        gini_scores = pd.read_csv(gini_scores_path)
        print(f"Scores de Gini chargés depuis : {gini_scores_path}")
        return gini_scores
    else:
        print(f"Fichier de scores de Gini introuvable : {gini_scores_path}")
        return None

