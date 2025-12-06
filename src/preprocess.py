import os
import pandas as pd
import numpy as np

def yyqq_to_date(yyqq):
    yy = int(yyqq[1:3])
    qq = yyqq[3:5]
    year = 2000 + yy if yy < 50 else 1900 + yy
    month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[qq]
    return pd.Timestamp(year=year, month=month, day=1)

def create_loan_date_column(data):
    data["Origination_date"] = data["Loanref"].apply(yyqq_to_date)
    return data

def to_float64(data):
    exclude = ["Origination_date"]

    for col in data.columns:
        if col not in exclude:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

    return data

def round_float64(data, n_decimales=4):
    float_cols = data.select_dtypes(include='float64').columns
    data[float_cols] = data[float_cols].round(n_decimales)
    return data

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

def read_sas_by_chunks(file_path, chunk_size=50000):
    """
    Lit un fichier SAS et le divise en chunks pour gérer les gros fichiers.
    Note: pyreadstat lit le fichier complet, puis on le divise en chunks en mémoire.
    
    Args:
        file_path: Chemin vers le fichier .sas7bdat
        chunk_size: Nombre de lignes par chunk
    
    Yields:
        DataFrame: Chunks du fichier SAS
    """
    try:
        import pyreadstat
    except ImportError:
        raise ImportError("pyreadstat est requis pour lire les fichiers SAS. Installez-le avec: conda install -c conda-forge pyreadstat")
    
    # Lire le fichier SAS avec pyreadstat (plus efficace que pandas pour les gros fichiers)
    print(f"  Lecture du fichier SAS avec pyreadstat...")
    try:
        df, meta = pyreadstat.read_sas7bdat(file_path)
        total_rows = len(df)
        print(f"  Fichier SAS lu: {total_rows} lignes, {len(df.columns)} colonnes")
    except MemoryError:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  ⚠ ERREUR: Mémoire insuffisante pour lire le fichier SAS ({file_size_mb:.2f} MB)")
        print(f"  Le fichier est trop volumineux pour être chargé en mémoire.")
        print(f"  SOLUTION: Convertir le fichier SAS en CSV d'abord avec SAS ou un autre outil.")
        raise MemoryError(f"Fichier SAS trop volumineux ({file_size_mb:.2f} MB) pour être chargé en mémoire")
    except Exception as e:
        print(f"  ⚠ ERREUR lors de la lecture du fichier SAS: {str(e)}")
        raise
    
    # Diviser le DataFrame en chunks
    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # Arrondi supérieur
    
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        if len(chunk) > 0:
            yield chunk

def impute_missing_data(data):
    ## TODO Analyser la pertinence d'utiliser une autre méthode d'imputation
    cleaned_df = data.dropna()
    return cleaned_df

def preprocess(data):
    keep_colnames = [
        "Loanref", "Credit_Score", "Mortgage_Insurance", "Number_of_units",
        "CLoan_to_value", "Debt_to_income", "OLoan_to_value",
        "Single_borrower",
        "is_Loan_purpose_purc", "is_Loan_purpose_cash", "is_Loan_purpose_noca",
        "is_First_time_homeowner", "is_First_time_homeowner_No",
        "is_Occupancy_status_prim", "is_Occupancy_status_inve", "is_Occupancy_status_seco",
        "is_Origination_channel_reta", "is_Origination_channel_brok", "is_Origination_channel_corr", "is_Origination_channel_tpo",
        "is_Property_type_cond", "is_Property_type_coop", "is_Property_type_manu",
        "is_Property_type_pud", "is_Property_type_sing",
        "DFlag"
    ]
    data = data[keep_colnames]
    data = data.copy()
    data = create_loan_date_column(data)
    data = data.sort_values("Origination_date")
    data = data.drop(columns=['Loanref'])

    data = to_float64(data)

    for col in ["CLoan_to_value", "OLoan_to_value"]:
        if col in data.columns:
            data[col] = data[col] / 100.0

    data = round_float64(data)

    data = impute_missing_data(data)
    
    # Sélection de variables basée sur le coefficient de Gini
    # Garder les 20 meilleures variables (ou toutes si moins de 20)
    data, gini_scores = select_features_by_gini(data, target_col='DFlag', min_gini_gain=0.0, top_k=20)

    return data

def process_and_save_all(project_path, windows=["FM12", "FM24", "FM36", "FM48", "FM60"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"]):
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
                    print(f"Traitement : {raw_path}")
                    df = pd.read_csv(raw_path)
                    df_processed = preprocess(df)
                    
                    # Récupérer les scores de Gini depuis la fonction preprocess modifiée
                    # On doit recalculer pour obtenir les scores
                    keep_colnames = [
                        "Loanref", "Credit_Score", "Mortgage_Insurance", "Number_of_units",
                        "CLoan_to_value", "Debt_to_income", "OLoan_to_value",
                        "Single_borrower",
                        "is_Loan_purpose_purc", "is_Loan_purpose_cash", "is_Loan_purpose_noca",
                        "is_First_time_homeowner", "is_First_time_homeowner_No",
                        "is_Occupancy_status_prim", "is_Occupancy_status_inve", "is_Occupancy_status_seco",
                        "is_Origination_channel_reta", "is_Origination_channel_brok", "is_Origination_channel_corr", "is_Origination_channel_tpo",
                        "is_Property_type_cond", "is_Property_type_coop", "is_Property_type_manu",
                        "is_Property_type_pud", "is_Property_type_sing",
                        "DFlag"
                    ]
                    df_temp = df[keep_colnames].copy()
                    df_temp = create_loan_date_column(df_temp)
                    df_temp = df_temp.sort_values("Origination_date")
                    df_temp = df_temp.drop(columns=['Loanref'])
                    df_temp = to_float64(df_temp)
                    
                    for col in ["CLoan_to_value", "OLoan_to_value"]:
                        if col in df_temp.columns:
                            df_temp[col] = df_temp[col] / 100.0
                    
                    df_temp = round_float64(df_temp)
                    df_temp = impute_missing_data(df_temp)
                    
                    # Calculer et sauvegarder les scores de Gini
                    gini_scores_dir = os.path.join(project_path, "outputs", "gini_scores", window, segment)
                    gini_scores_path = os.path.join(gini_scores_dir, f"gini_scores_{window}.csv")
                    df_processed, gini_scores = select_features_by_gini(
                        df_temp, target_col='DFlag', min_gini_gain=0.0, top_k=20, 
                        save_scores_path=gini_scores_path
                    )
                    selected_features = df_processed.columns.tolist()
                    
                    save_dir = os.path.join(project_path, "data", "processed", window, segment)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"train_{window[2:]}.csv")
                    df_processed.to_csv(save_path, index=False)
                    print(f"Sauvegardé : {save_path}")
                else:
                    print(f"Fichier introuvable : {raw_path}")
            
            # Ensuite, traiter les autres splits en utilisant les mêmes variables que train
            for split in splits:
                if split == "train":
                    continue  # Déjà traité
                    
                filename = f"{split}_{window[2:]}.csv"
                if split == "OOU":
                    filename = f"{split}.sas7bdat"
                raw_path = os.path.join(project_path, "data", "raw", window, segment, filename)
                if os.path.exists(raw_path):
                    print(f"Traitement : {raw_path}")
                    if split == "OOU":
                        # Traiter le fichier SAS par chunks
                        print(f"  Lecture du fichier SAS par chunks...")
                        chunk_size = 50000  # Nombre de lignes par chunk
                        
                        # Créer le dossier de destination
                        save_dir = os.path.join(project_path, "data", "processed", window, segment)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{split}_{window[2:]}.csv")
                        
                        # Lire et traiter par chunks
                        first_chunk = True
                        chunk_num = 1
                        total_rows = 0
                        
                        try:
                            for df_chunk in read_sas_by_chunks(raw_path, chunk_size=chunk_size):
                                if len(df_chunk) == 0:
                                    break
                                
                                print(f"  Traitement du chunk {chunk_num} ({len(df_chunk)} lignes)...")
                                
                                # Utiliser le même preprocessing mais avec les variables sélectionnées depuis train
                                keep_colnames = [
                                    "Loanref", "Credit_Score", "Mortgage_Insurance", "Number_of_units",
                                    "CLoan_to_value", "Debt_to_income", "OLoan_to_value",
                                    "Single_borrower",
                                    "is_Loan_purpose_purc", "is_Loan_purpose_cash", "is_Loan_purpose_noca",
                                    "is_First_time_homeowner", "is_First_time_homeowner_No",
                                    "is_Occupancy_status_prim", "is_Occupancy_status_inve", "is_Occupancy_status_seco",
                                    "is_Origination_channel_reta", "is_Origination_channel_brok", "is_Origination_channel_corr", "is_Origination_channel_tpo",
                                    "is_Property_type_cond", "is_Property_type_coop", "is_Property_type_manu",
                                    "is_Property_type_pud", "is_Property_type_sing",
                                    "DFlag"
                                ]
                                # Filtrer pour ne garder que les colonnes qui existent dans le DataFrame
                                available_colnames = [col for col in keep_colnames if col in df_chunk.columns]
                                if chunk_num == 1 and len(available_colnames) < len(keep_colnames):
                                    missing_colnames = [col for col in keep_colnames if col not in df_chunk.columns]
                                    print(f"  Attention: Colonnes manquantes dans le fichier: {missing_colnames}")
                                
                                df_temp = df_chunk[available_colnames].copy()
                                df_temp = create_loan_date_column(df_temp)
                                df_temp = df_temp.sort_values("Origination_date")
                                df_temp = df_temp.drop(columns=['Loanref'])
                                df_temp = to_float64(df_temp)
                                
                                for col in ["CLoan_to_value", "OLoan_to_value"]:
                                    if col in df_temp.columns:
                                        df_temp[col] = df_temp[col] / 100.0
                                
                                df_temp = round_float64(df_temp)
                                df_temp = impute_missing_data(df_temp)
                                
                                # Utiliser les mêmes variables que train
                                if selected_features is not None:
                                    # S'assurer que toutes les variables sélectionnées existent
                                    available_features = [f for f in selected_features if f in df_temp.columns]
                                    chunk_processed = df_temp[available_features]
                                else:
                                    # Si train n'a pas été traité, utiliser preprocess normal
                                    chunk_processed = preprocess(df_chunk)
                                
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
                            
                            print(f"  Sauvegardé : {save_path} ({total_rows} lignes au total)")
                            
                        except ImportError as e:
                            print(f"\n  ⚠ ERREUR: pyreadstat n'est pas installé.")
                            file_size_mb = os.path.getsize(raw_path) / (1024 * 1024)
                            print(f"  INFO: Taille du fichier: {file_size_mb:.2f} MB")
                            print(f"\n  SOLUTIONS POSSIBLES:")
                            print(f"  1. [RECOMMANDÉ] Installer pyreadstat avec conda (précompilé):")
                            print(f"     conda install -c conda-forge pyreadstat")
                            print(f"  2. Installer Microsoft C++ Build Tools puis: pip install pyreadstat")
                            print(f"     Télécharger depuis: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
                            raise
                        except Exception as e:
                            print(f"\n  ⚠ ERREUR lors du traitement du fichier SAS: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            raise
                    else:
                        # Pour les fichiers CSV, lire normalement ou par chunks selon la taille
                        df = pd.read_csv(raw_path)
                        
                        # Utiliser le même preprocessing mais avec les variables sélectionnées depuis train
                        keep_colnames = [
                            "Loanref", "Credit_Score", "Mortgage_Insurance", "Number_of_units",
                            "CLoan_to_value", "Debt_to_income", "OLoan_to_value",
                            "Single_borrower",
                            "is_Loan_purpose_purc", "is_Loan_purpose_cash", "is_Loan_purpose_noca",
                            "is_First_time_homeowner", "is_First_time_homeowner_No",
                            "is_Occupancy_status_prim", "is_Occupancy_status_inve", "is_Occupancy_status_seco",
                            "is_Origination_channel_reta", "is_Origination_channel_brok", "is_Origination_channel_corr", "is_Origination_channel_tpo",
                            "is_Property_type_cond", "is_Property_type_coop", "is_Property_type_manu",
                            "is_Property_type_pud", "is_Property_type_sing",
                            "DFlag"
                        ]
                        # Filtrer pour ne garder que les colonnes qui existent dans le DataFrame
                        available_colnames = [col for col in keep_colnames if col in df.columns]
                        missing_colnames = [col for col in keep_colnames if col not in df.columns]
                        if missing_colnames:
                            print(f"  Attention: Colonnes manquantes dans le fichier: {missing_colnames}")
                        df_temp = df[available_colnames].copy()
                        df_temp = create_loan_date_column(df_temp)
                        df_temp = df_temp.sort_values("Origination_date")
                        df_temp = df_temp.drop(columns=['Loanref'])
                        df_temp = to_float64(df_temp)
                        
                        for col in ["CLoan_to_value", "OLoan_to_value"]:
                            if col in df_temp.columns:
                                df_temp[col] = df_temp[col] / 100.0
                        
                        df_temp = round_float64(df_temp)
                        df_temp = impute_missing_data(df_temp)
                        
                        # Utiliser les mêmes variables que train
                        if selected_features is not None:
                            # S'assurer que toutes les variables sélectionnées existent
                            available_features = [f for f in selected_features if f in df_temp.columns]
                            df_processed = df_temp[available_features]
                        else:
                            # Si train n'a pas été traité, utiliser preprocess normal
                            df_processed = preprocess(df)

                        save_dir = os.path.join(project_path, "data", "processed", window, segment)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{split}_{window[2:]}.csv")

                        df_processed.to_csv(save_path, index=False)
                        print(f"Sauvegardé : {save_path}")
                else:
                    print(f"Fichier introuvable : {raw_path}")

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

def load_processed_data(project_path, windows=["FM12", "FM24", "FM36", "FM48", "FM60"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"]):
    dataframes = []
    for window in windows:
        for segment in segments:
            for split in splits:
                filename = f"{split}_{window[2:]}.csv"
                file_path = os.path.join(project_path, "data", "processed", window, segment, filename)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
                else:
                    print(f"Fichier introuvable : {file_path}")
    return pd.concat(dataframes, ignore_index=True)