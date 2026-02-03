"""
================================================================================
PARTIE 1 : ANALYSE EXPLORATOIRE & COMPRÉHENSION DES DONNÉES
================================================================================
Objectif : Vérifier les dimensions du dataset brut et identifier les features.
Ce script aide à comprendre la structure des données avant tout traitement.
"""
import pandas as pd
import numpy as np
import torch
import yaml

# Charger les données
df = pd.read_excel('datasets/raw/dataset-uci.xlsx', sheet_name='Dataset')

# Supprimer la colonne d'index
if df.columns[0] == 'Unnamed: 0':
    df = df.drop(columns=[df.columns[0]])

print("=== ANALYSE DU DATASET ===")
print(f"Shape original: {df.shape}")
print(f"Colonnes: {list(df.columns)}")

# Compter les features
target_col = 'Case Type'
features = [col for col in df.columns if col != target_col]
print(f"\nNombre de features originales: {len(features)}")

# Calculer les features après prétraitement
# Original: features de base
# Ajout: Parent_Age_Diff, Diagnosis_Ratio (+2)
# Total: len(features) + 2
total_features = len(features) + 2
print(f"Nombre de features après prétraitement: {total_features}")

# Vérifier avec config.yml
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

print(f"\n=== CONFIGURATION ===")
print(f"input_size dans config.yml: {config['model']['input_size']}")
print(f"input_size devrait être: {total_features}")

# Mettre à jour config.yml si nécessaire
if config['model']['input_size'] != total_features:
    print(f"\n  ATTENTION: input_size incorrect!")
    print(f"   Config: {config['model']['input_size']}")
    print(f"   Réel: {total_features}")
    
    # Mettre à jour la config
    config['model']['input_size'] = total_features
    with open('config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f" config.yml mis à jour avec input_size: {total_features}")