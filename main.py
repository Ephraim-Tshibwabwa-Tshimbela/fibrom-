#!/usr/bin/env python3
"""
Script principal pour le projet de classification NF1
TP Deep Learning - Bac4 Génie Logiciel
"""

import argparse
import sys
import os
import torch

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='Projet de classification NF1')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'test', 'all'],
                       default='all', help='Mode d\'exécution')
    parser.add_argument('--config', default='config.yml',
                       help='Chemin vers le fichier de configuration')
    
    args = parser.parse_args()
    
    if args.mode in ['preprocess', 'all']:
        print("\n" + "="*60)
        print("PHASE 1: PRÉTRAITEMENT DES DONNÉES")
        print("="*60)
        
        # Vérifier si les données sont déjà prétraitées
        if os.path.exists('datasets/processed/train_data.pt'):
            print("Les données sont déjà prétraitées. Passage à l'entraînement...")
        else:
            try:
                from src.dataset.preprocessor import NF1DatasetPreprocessor
                preprocessor = NF1DatasetPreprocessor(args.config)
                datasets, scaler = preprocessor.process_pipeline()
            except AttributeError as e:
                print(f"Erreur: {e}")
                print("Le prétraitement a échoué. Utilisation des données existantes...")
    
    if args.mode in ['train', 'all']:
        print("\n" + "="*60)
        print("PHASE 2: ENTRAÎNEMENT DU MODÈLE")
        print("="*60)
        
        # Vérifier si les données existent
        if not os.path.exists('datasets/processed/train_data.pt'):
            print("Erreur: Les données prétraitées n'existent pas.")
            print("Exécutez d'abord: python main.py --mode preprocess")
            return
        
        from src.train import main as train_main
        
        history = train_main()
    
    if args.mode in ['test', 'all']:
        print("\n" + "="*60)
        print("PHASE 3: ÉVALUATION DU MODÈLE")
        print("="*60)
        
        # Vérifier si le modèle existe
        if not os.path.exists('models/nf1_classifier.pth'):
            print("Erreur: Le modèle entraîné n'existe pas.")
            print("Exécutez d'abord: python main.py --mode train")
            return
        
        from src.test import main as test_main
        
        summary, importance_df = test_main()
    
    print("\n" + "="*60)
    print("EXÉCUTION TERMINÉE AVEC SUCCÈS")
    print("="*60)

if __name__ == "__main__":
    main()