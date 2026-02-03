#!/usr/bin/env python3
"""
Script pour corriger le fichier train.py
"""

import os

# Lire le contenu actuel de train.py
with open('src/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Trouver la définition de la classe Trainer
lines = content.split('\n')

# Rechercher la ligne où le modèle est créé
model_creation_line = None
for i, line in enumerate(lines):
    if 'self.model =' in line and 'NF1Classifier' in line:
        model_creation_line = i
        break

if model_creation_line is not None:
    print("Le modèle est déjà créé directement. Vérification de la méthode create_model...")
    
    # Vérifier si create_model existe
    if 'def create_model' in content:
        print("La méthode create_model existe déjà.")
    else:
        print("La méthode create_model n'existe pas. Ajout...")
        
        # Trouver l'endroit pour ajouter la méthode
        # Chercher après __init__ et avant les autres méthodes
        for i, line in enumerate(lines):
            if 'def __init__' in line:
                init_start = i
                break
        
        # Trouver la fin de __init__
        for i in range(init_start, len(lines)):
            if lines[i].strip() == '' and i > init_start + 20:  # Fin approximative
                insert_point = i
                break
        else:
            insert_point = init_start + 30
        
        # Méthode create_model à ajouter
        create_model_method = '''
    def create_model(self):
        """Créer le modèle avec la bonne taille d'entrée"""
        return NF1Classifier(
            input_size=self.config['model']['input_size'],
            hidden_layers=self.config['model']['hidden_layers'],
            dropout_rate=self.config['model']['dropout_rate'],
            activation=self.config['model']['activation']
        ).to(self.device)'''
        
        # Insérer la méthode
        lines.insert(insert_point, create_model_method)
        
        # Réécrire le fichier
        with open('src/train.py', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print("✅ Méthode create_model ajoutée.")
else:
    print("Le modèle n'est pas créé directement. Correction de l'initialisation...")
    
    # Chercher la ligne où input_size est détecté
    for i, line in enumerate(lines):
        if 'Taille d\'entrée détectée:' in line or 'input_size' in line:
            # Remplacer la création du modèle
            new_model_creation = '''        # Initialiser le modèle
        self.model = NF1Classifier(
            input_size=self.config['model']['input_size'],
            hidden_layers=self.config['model']['hidden_layers'],
            dropout_rate=self.config['model']['dropout_rate'],
            activation=self.config['model']['activation']
        ).to(self.device)'''
            
            # Trouver la ligne self.model = NF1Classifier(config_path).to(self.device)
            for j in range(i, len(lines)):
                if 'self.model = NF1Classifier' in lines[j]:
                    lines[j] = new_model_creation
                    break
            
            # Réécrire le fichier
            with open('src/train.py', 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print("✅ Création du modèle corrigée.")
            break

print("\nPour exécuter l'entraînement :")
print("python main.py --mode train")