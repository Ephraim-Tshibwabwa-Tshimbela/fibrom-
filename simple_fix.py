#!/usr/bin/env python3
"""
Script simplifié pour corriger tous les problèmes
"""

import os
import shutil
import pandas as pd
import numpy as np

print("=== CORRECTION SIMPLIFIÉE ===")

# 1. Nettoyer les dossiers
print("\n1. Nettoyage des dossiers...")
folders_to_clean = ['datasets/processed', 'models', 'results']
for folder in folders_to_clean:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"  Supprimé: {folder}")
    os.makedirs(folder, exist_ok=True)
    print(f"  Créé: {folder}")

# 2. Créer un fichier config.yml propre
print("\n2. Création d'un nouveau config.yml...")
config_content = """data:
  data_path: "datasets/raw/dataset-uci.xlsx"
  sheet_name: "Dataset"
  test_size: 0.15
  val_size: 0.15
  random_state: 42

model:
  input_size: 21
  hidden_layers: [64, 32, 16]
  dropout_rate: 0.3
  activation: "relu"

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  patience: 10
  device: "cpu"

paths:
  processed_data: "datasets/processed/"
  model_save: "models/nf1_classifier.pth"
  results: "results/"
"""

with open('config.yml', 'w', encoding='utf-8') as f:
    f.write(config_content)
print("✅ config.yml créé avec encodage UTF-8")

# 3. Analyser le dataset
print("\n3. Analyse du dataset...")
df = pd.read_excel('datasets/raw/dataset-uci.xlsx', sheet_name='Dataset')
print(f"  Shape original: {df.shape}")

# Supprimer la colonne d'index
if df.columns[0] == 'Unnamed: 0':
    df = df.drop(columns=['Unnamed: 0'])
    print(f"  Colonne 'Unnamed: 0' supprimée")

# Renommer Learning Disability
if 'Learninn Disability' in df.columns:
    df = df.rename(columns={'Learninn Disability': 'Learning Disability'})

print(f"  Shape après nettoyage: {df.shape}")
print(f"  Colonnes: {list(df.columns)}")

# 4. Prétraiter les données manuellement
print("\n4. Prétraitement manuel des données...")

# Gestion des valeurs manquantes
numeric_cols = ['Age of Mother', 'Age of Father', 'Age at First Diagnosis']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Feature engineering
df['Parent_Age_Diff'] = abs(df['Age of Father'] - df['Age of Mother'])
df['Diagnosis_Ratio'] = df['Age at First Diagnosis'] / (df['Age of Mother'] + 1e-6)

# Séparer features et target
X = df.drop(columns=['Case Type'])
y = df['Case Type']

print(f"  Nombre de features: {X.shape[1]}")
print(f"  Nombre d'échantillons: {X.shape[0]}")

# 5. Diviser les données
print("\n5. Division train/val/test...")
from sklearn.model_selection import train_test_split

# Split train + (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Split val/test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"  Train: {X_train.shape[0]} échantillons")
print(f"  Validation: {X_val.shape[0]} échantillons")
print(f"  Test: {X_test.shape[0]} échantillons")

# 6. Normaliser
print("\n6. Normalisation des données...")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 7. Sauvegarder en format PyTorch
print("\n7. Sauvegarde des données prétraitées...")
import torch

train_data = {
    'X': torch.FloatTensor(X_train_scaled),
    'y': torch.FloatTensor(y_train.values).unsqueeze(1)
}

val_data = {
    'X': torch.FloatTensor(X_val_scaled),
    'y': torch.FloatTensor(y_val.values).unsqueeze(1)
}

test_data = {
    'X': torch.FloatTensor(X_test_scaled),
    'y': torch.FloatTensor(y_test.values).unsqueeze(1)
}

torch.save(train_data, 'datasets/processed/train_data.pt')
torch.save(val_data, 'datasets/processed/val_data.pt')
torch.save(test_data, 'datasets/processed/test_data.pt')

print(f"✅ Données sauvegardées dans datasets/processed/")
print(f"   Taille des tensors: {train_data['X'].shape}")

print("\n=== CORRECTION TERMINÉE ===")
print("\nPour exécuter l'entraînement :")
print("python main.py --mode train")