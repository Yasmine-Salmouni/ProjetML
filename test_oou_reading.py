"""
Script utilitaire pour tester la lecture des fichiers OOU (SAS)
Permet de diagnostiquer les problèmes de lecture des fichiers .sas7bdat
"""

import os
import sys

PROJECT_PATH = os.getcwd()
SRC_PATH = os.path.join(PROJECT_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from preprocess_xgboost import read_sas_file

def test_oou_files(project_path, window="FM12", segments=["green", "red"]):
    """
    Teste la lecture de tous les fichiers OOU pour diagnostiquer les problèmes.
    
    Args:
        project_path: Chemin du projet
        window: Fenêtre temporelle (ex: "FM12")
        segments: Liste des segments (ex: ["green", "red"])
    """
    print("="*80)
    print("TEST DE LECTURE DES FICHIERS OOU (SAS)")
    print("="*80)
    print(f"Chemin du projet : {project_path}")
    print(f"Fenêtre : {window}")
    print(f"Segments : {segments}")
    print()
    
    # Vérifier les dépendances
    print("1. Vérification des dépendances...")
    dependencies = {
        'sas7bdat': False,
        'pandas': False
    }
    
    try:
        from sas7bdat import SAS7BDAT
        dependencies['sas7bdat'] = True
        print("  sas7bdat installé (nécessaire pour lire les fichiers SAS)")
    except ImportError:
        print("  sas7bdat non installé - installer avec: pip install sas7bdat")
        print("     sas7bdat est OBLIGATOIRE pour lire les fichiers SAS")
    
    try:
        import pandas as pd
        dependencies['pandas'] = True
        print("  pandas installé (nécessaire pour les DataFrames)")
    except ImportError:
        print("  pandas non installé")
    
    print()
    
    # Tester la lecture de chaque fichier OOU
    print("2. Test de lecture des fichiers OOU...")
    print()
    
    for segment in segments:
        filename = "OOU.sas7bdat"
        file_path = os.path.join(project_path, "data", "raw", window, segment, filename)
        
        print(f"  Segment : {segment}")
        print(f"  Fichier : {file_path}")
        
        if not os.path.exists(file_path):
            print(f"  Fichier introuvable")
            print()
            continue
        
        # Afficher la taille du fichier
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"  Taille : {file_size_mb:.2f} MB ({file_size:,} bytes)")
        
        # Essayer de lire le fichier par chunks
        try:
            print(f"  Tentative de lecture par chunks...")
            chunk_size = 50000
            chunk_generator = read_sas_file(file_path, chunk_size=chunk_size)
            
            # Lire le premier chunk pour obtenir les informations
            first_chunk = next(chunk_generator)
            
            print(f"  Premier chunk lu avec succès!")
            print(f"     - Taille du chunk : {len(first_chunk):,} lignes")
            print(f"     - Nombre de colonnes : {len(first_chunk.columns)}")
            print(f"     - Colonnes : {list(first_chunk.columns)[:10]}..." if len(first_chunk.columns) > 10 else f"     - Colonnes : {list(first_chunk.columns)}")
            
            # Vérifier si DFlag est présent
            if 'DFlag' in first_chunk.columns:
                print(f"     - Colonne DFlag présente")
                # Compter les lignes totales et la distribution DFlag
                total_rows = len(first_chunk)
                dflag_counts = first_chunk['DFlag'].value_counts().to_dict()
                
                # Lire les chunks restants pour obtenir le total
                chunk_count = 1
                for chunk in chunk_generator:
                    total_rows += len(chunk)
                    chunk_count += 1
                    # Mettre à jour la distribution DFlag
                    chunk_dflag = chunk['DFlag'].value_counts().to_dict()
                    for key, value in chunk_dflag.items():
                        dflag_counts[key] = dflag_counts.get(key, 0) + value
                
                print(f"     - Nombre total de lignes : {total_rows:,}")
                print(f"     - Nombre de chunks : {chunk_count}")
                print(f"     - Distribution DFlag : {dflag_counts}")
            else:
                print(f"     - Colonne DFlag absente")
                print(f"     - Colonnes disponibles : {list(first_chunk.columns)}")
                # Compter quand même les lignes totales
                total_rows = len(first_chunk)
                chunk_count = 1
                for chunk in chunk_generator:
                    total_rows += len(chunk)
                    chunk_count += 1
                print(f"     - Nombre total de lignes : {total_rows:,}")
                print(f"     - Nombre de chunks : {chunk_count}")
            
            # Afficher les types de données (du premier chunk)
            print(f"     - Types de données (échantillon du premier chunk) :")
            type_counts = first_chunk.dtypes.value_counts()
            for dtype, count in type_counts.items():
                print(f"       {dtype}: {count} colonnes")
            
        except StopIteration:
            print(f"  Le fichier semble vide (aucun chunk)")
        except Exception as e:
            print(f"  Erreur lors de la lecture : {str(e)}")
            print(f"     Type d'erreur : {type(e).__name__}")
            import traceback
            print(f"     Détails :")
            traceback.print_exc()
        
        print()
    
    print("="*80)
    print("TEST TERMINÉ")
    print("="*80)
    
    # Recommandations
    print("\nRecommandations :")
    if not dependencies['sas7bdat']:
        print("  - Installer sas7bdat (OBLIGATOIRE) : pip install sas7bdat")
        print("     sas7bdat est la seule méthode utilisée pour lire les fichiers SAS")
    print("  - Si les fichiers sont très volumineux, vérifier la mémoire disponible")
    print("  - Si sas7bdat échoue, convertir les fichiers SAS en CSV avec un autre outil (SAS, R, etc.)")

if __name__ == "__main__":
    test_oou_files(
        project_path=PROJECT_PATH,
        window="FM12",
        segments=["green", "red"]
    )

