#!/usr/bin/env python3
"""
Script pour corriger tous les problèmes
"""

import os
import shutil
import pandas as pd
import yaml

print("=== DÉBUT DE LA CORRECTION ===")

# 1. Nettoyer les dossiers
print("\n1. Nettoyage des dossiers...")
folders_to_clean = ['datasets/processed', 'models', 'results']
for folder in folders_to_clean:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"  Supprimé: {folder}")
    os.makedirs(folder, exist_ok=True)
    print(f"  Créé: {folder}")

# 2. Analyser le dataset original
print("\n2. Analyse du dataset original...")
df = pd.read_excel('datasets/raw/dataset-uci.xlsx', sheet_name='Dataset')
print(f"  Shape: {df.shape}")
print(f"  Colonnes: {list(df.columns)}")

# Compter les features
n_features = len(df.columns) - 2  # -1 pour Unnamed: 0, -1 pour Case Type
print(f"  Features de base: {n_features}")

# 3. Mettre à jour config.yml
print("\n3. Mise à jour de config.yml...")
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Calculer la bonne taille
# Features de base + 2 nouvelles features = n_features + 2
correct_input_size = n_features + 2
config['model']['input_size'] = correct_input_size

# Forcer CPU pour éviter les problèmes GPU
config['training']['device'] = 'cpu'

with open('config.yml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"  input_size mis à jour: {correct_input_size}")
print(f"  device mis à jour: cpu")

# 4. Corriger le prétraitement
print("\n4. Correction du prétraitement...")
preprocessor_path = 'src/dataset/preprocessor.py'

with open(preprocessor_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Remplacer la fonction load_raw_data
new_load_raw_data = '''
def load_raw_data(self):
    """Charger les données brutes depuis Excel"""
    df = pd.read_excel(
        self.data_path, 
        sheet_name=self.config['data']['sheet_name'],
        header=0
    )
    
    # SUPPRIMER LA PREMIÈRE COLONNE SI C'EST UN INDEX
    # (peut être 'Unnamed: 0' ou première colonne vide)
    first_col = df.columns[0]
    if first_col == 'Unnamed: 0' or pd.isna(first_col) or first_col == '':
        df = df.drop(columns=[first_col])
        print(f"  Colonne '{first_col}' supprimée")
    
    # Renommer les colonnes pour uniformité
    if 'Learninn Disability' in df.columns:
        df = df.rename(columns={'Learninn Disability': 'Learning Disability'})
    
    print(f" Dataset chargé - Shape: {df.shape}")
    print(f"   Features: {len(df.columns) - 1} (sans la target)")
    
    return df
'''

# Trouver et remplacer la fonction
lines = content.split('\n')
new_lines = []
i = 0
while i < len(lines):
    if 'def load_raw_data' in lines[i]:
        # Ajouter la nouvelle fonction
        new_lines.append(new_load_raw_data)
        # Sauter jusqu'à la fin de l'ancienne fonction
        while i < len(lines) and not lines[i].strip().startswith('def ') and i > 0:
            i += 1
        i -= 1
    else:
        new_lines.append(lines[i])
    i += 1

with open(preprocessor_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))

print("✅ Prétraitement corrigé")

# 5. Exécuter le prétraitement
print("\n5. Exécution du prétraitement...")
import sys
sys.path.append('src')

from dataset.preprocessor import NF1DatasetPreprocessor

preprocessor = NF1DatasetPreprocessor()
datasets, scaler = preprocessor.process_pipeline()

print("\n=== CORRECTION TERMINÉE ===")
print("\nPour exécuter l'entraînement :")
print("python main.py --mode train")
print("\nPour exécuter le test :")
print("python main.py --mode test")