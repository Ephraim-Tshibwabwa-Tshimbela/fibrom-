"""
================================================================================
PARTIE 1 : ANALYSE EXPLORATOIRE & COMPRÉHENSION DES DONNÉES
================================================================================
Objectif : Analyser le contenu des données (types, valeurs manquantes, distribution).
Sert à justifier les choix de prétraitement (Partie 2).
"""
import pandas as pd
import numpy as np
import torch

# 1. Vérifier le dataset original
print("=== VÉRIFICATION DU DATASET ===")
df = pd.read_excel('datasets/raw/dataset-uci.xlsx', sheet_name='Dataset')
print(f"Shape du dataframe: {df.shape}")
print(f"Colonnes: {df.columns.tolist()}")
print(f"Nombre de colonnes: {len(df.columns)}")
print(f"Colonnes avec 'Unnamed': {[col for col in df.columns if 'Unnamed' in col]}")

# 2. Vérifier les données prétraitées
print("\n=== VÉRIFICATION DES DONNÉES PRÉTRAITÉES ===")
try:
    train_data = torch.load('datasets/processed/train_data.pt')
    print(f"Shape des features train: {train_data['X'].shape}")
    print(f"Shape des labels train: {train_data['y'].shape}")
    print(f"Nombre de features: {train_data['X'].shape[1]}")
except Exception as e:
    print(f"Erreur lors du chargement: {e}")

# 3. Vérifier la configuration
import yaml
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
print(f"\n=== CONFIGURATION ===")
print(f"input_size dans config: {config['model']['input_size']}")