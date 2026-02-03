"""
================================================================================
PARTIE 3 : ARCHITECTURE DU MODÈLE (NEURAL NETWORK)
================================================================================
Objectif : Définir l'architecture du Perceptron Multicouche (MLP).
Composants : Couches Linéaires, ReLU, BatchNorm, Dropout.
Note : Le Sigmoid final a été retiré pour utiliser BCEWithLogitsLoss (stabilisation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class NF1Classifier(nn.Module):
    """Réseau de neurones pour la classification NF1"""
    
    def __init__(self, input_size=20, hidden_layers=None, 
                 dropout_rate=0.3, activation='relu', config_path=None):
        """
        Args:
            input_size: Taille des features d'entrée (20 dans notre cas)
            hidden_layers: Liste des tailles des couches cachées
            dropout_rate: Taux de dropout
            activation: Fonction d'activation
            config_path: Optionnel - chemin vers fichier de config
        """
        super(NF1Classifier, self).__init__()
        
        # Si un fichier de config est fourni, le charger
        if config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            model_config = config['model']
            input_size = model_config['input_size']
            hidden_layers = model_config['hidden_layers']
            dropout_rate = model_config['dropout_rate']
            activation = model_config['activation']
        
        # Valeurs par défaut si hidden_layers n'est pas fourni
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        
        # Construire les couches
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Couche de sortie
        layers.append(nn.Linear(prev_size, 1))
        # layers.append(nn.Sigmoid())  # Retiré pour utiliser BCEWithLogitsLoss
        
        self.model = nn.Sequential(*layers)
        
        # Sauvegarder la configuration
        self.config = {
            'input_size': input_size,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'activation': activation
        }
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x, threshold=0.5):
        """Prédiction avec seuil"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()
        return predictions, probabilities