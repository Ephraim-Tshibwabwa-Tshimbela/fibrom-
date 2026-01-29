"""
================================================================================
PARTIE 2 : PIPELINE DE PRÉTRAITEMENT (ETL)
================================================================================
Objectif : Transformer les données brutes en tenseurs propres pour le modèle.
Responsabilités : Nettoyage, Encodage, Normalisation (StandardScaler), Split Train/Test.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import os

class NF1DatasetPreprocessor:
    """Classe pour le prétraitement des données NF1"""
    
    def __init__(self, config_path="config.yml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = self.config['data']['data_path']
        self.test_size = self.config['data']['test_size']
        self.val_size = self.config['data']['val_size']
        self.random_state = self.config['data']['random_state']
        
    def process_pipeline(self):
        """Pipeline complet de prétraitement"""
        print("Chargement des données brutes...")
        df = self.load_raw_data()
        
        print("Prétraitement des données...")
        X, y = self.preprocess_data(df)
        
        print("Division train/val/test...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        print("Normalisation des données...")
        X_train_norm, X_val_norm, X_test_norm, scaler = self.normalize_data(
            X_train, X_val, X_test
        )
        
        # Convertir en tensors PyTorch
        train_dataset = {
            'X': torch.FloatTensor(X_train_norm.values),
            'y': torch.FloatTensor(y_train.values).unsqueeze(1)
        }
        
        val_dataset = {
            'X': torch.FloatTensor(X_val_norm.values),
            'y': torch.FloatTensor(y_val.values).unsqueeze(1)
        }
        
        test_dataset = {
            'X': torch.FloatTensor(X_test_norm.values),
            'y': torch.FloatTensor(y_test.values).unsqueeze(1)
        }
        
        # Sauvegarder
        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        self.save_processed_data(datasets)
        
        print(f"Taille des datasets:")
        print(f"  Train: {len(train_dataset['X'])} échantillons")
        print(f"  Validation: {len(val_dataset['X'])} échantillons")
        print(f"  Test: {len(test_dataset['X'])} échantillons")
        
        return datasets, scaler        
        
    def load_raw_data(self):
        """Charger les données brutes depuis Excel"""
        df = pd.read_excel(
            self.data_path, 
            sheet_name=self.config['data']['sheet_name'],
            header=0
        )
        
        # SUPPRIMER LA COLONNE D'INDEX SI ELLE EXISTE
        # Plusieurs façons possibles
        if df.columns[0] == 'Unnamed: 0':
            df = df.drop(columns=[df.columns[0]])
        
        # OU: Supprimer toute colonne nommée 'Unnamed: 0'
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # Renommer les colonnes pour uniformité
        if 'Learninn Disability' in df.columns:
            df = df.rename(columns={'Learninn Disability': 'Learning Disability'})
        
        print(f" Dataset chargé - Shape: {df.shape}, Colonnes: {list(df.columns)}")
        return df
    
    def preprocess_data(self, df):
        """Prétraiter les données"""
        # Copie du dataframe
        df_processed = df.copy()
        
        # 1. Gestion des valeurs manquantes
        # Variables numériques - imputation par la médiane
        numeric_cols = ['Age of Mother', 'Age of Father', 'Age at First Diagnosis']
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # 2. Vérification des types
        # Toutes les variables binaires doivent être 0/1
        binary_cols = [
            'Tumour Case', 'Café au lait (CLS)', 'Axillary Freckles',
            'Inguinal Freckles', 'Lisch Nodules', 'Dermal Neurofibromins',
            'Plexiform Neurofibromins', 'Optic Glioma', 'Skeletal Dysplasia',
            'Learning Disability', 'Hypertension', 'Astrocytoma',
            'Hamartoma', 'Scoliosis', 'Other Symptoms'
        ]
        
        for col in binary_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(int)
        
        # 3. Feature engineering
        # Différence d'âge des parents
        df_processed['Parent_Age_Diff'] = abs(
            df_processed['Age of Father'] - df_processed['Age of Mother']
        )
        
        # Ratio d'âge au diagnostic
        df_processed['Diagnosis_Ratio'] = df_processed['Age at First Diagnosis'] / (
            df_processed['Age of Mother'] + 1e-6
        )
        
        # 4. Séparation features/target
        target_col = 'Case Type'
        if target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]
        else:
            raise ValueError(f"Colonne cible '{target_col}' non trouvée")
        
        return X, y
    
    def split_data(self, X, y):
        """Diviser les données en train/val/test"""
        # Split train + (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Split val/test
        val_ratio = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1-val_ratio,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_data(self, X_train, X_val, X_test):
        """Normaliser les données"""
        from sklearn.preprocessing import StandardScaler
        
        # Identifier les colonnes numériques
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Initialiser le scaler
        scaler = StandardScaler()
        
        # Normaliser les données
        X_train_norm = X_train.copy()
        X_val_norm = X_val.copy()
        X_test_norm = X_test.copy()
        
        X_train_norm[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val_norm[numeric_cols] = scaler.transform(X_val[numeric_cols])
        X_test_norm[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        return X_train_norm, X_val_norm, X_test_norm, scaler
    
    def save_processed_data(self, datasets, path=None):
        """Sauvegarder les données prétraitées"""
        if path is None:
            path = self.config['paths']['processed_data']
        
        os.makedirs(path, exist_ok=True)
        
        # Sauvegarder chaque dataset
        for name, dataset in datasets.items():
            torch.save(dataset, os.path.join(path, f'{name}_data.pt'))
        
        print(f"Données sauvegardées dans {path}")
    
