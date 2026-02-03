"""
================================================================================
PARTIE 4 : ENTRAÎNEMENT & OPTIMISATION
================================================================================
Objectif : Gérer la boucle d'apprentissage (Epochs, Batches, Backpropagation).
Points clés : Gestion du déséquilibre de classe (pos_weight), Early Stopping.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

from src.model.nf1_classifier import NF1Classifier
from src.dataset.loader import create_dataloaders

class Trainer:
    """Classe pour l'entraînement du modèle"""
    
    def __init__(self, config_path="config.yml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Configuration
        self.device = torch.device('cpu')  # Force CPU pour éviter les problèmes
        self.learning_rate = self.config['training']['learning_rate']
        self.epochs = self.config['training']['epochs']
        self.patience = self.config['training']['patience']
        
        # Créer les dataloaders
        self.train_loader, self.val_loader, self.test_loader, input_size = \
            create_dataloaders(config_path)
        
        # Mettre à jour la taille d'entrée
        self.config['model']['input_size'] = input_size
        
        print(f"Taille d'entrée: {input_size}")
        print(f"Device: {self.device}")
        
        # Initialiser le modèle
        self.model = NF1Classifier(
    input_size=input_size,  # ← Utiliser input_size calculé
    hidden_layers=self.config['model']['hidden_layers'],
    dropout_rate=self.config['model']['dropout_rate'],
    activation=self.config['model']['activation']
    ).to(self.device)
        
        # Calculer les poids de classe pour le rééquilibrage
        try:
            # Récupérer les labels du dataset d'entraînement
            # NF1Dataset stocke les données dans un dictionnaire self.data
            y_train = self.train_loader.dataset.data['y']
            num_pos = torch.sum(y_train).item()
            num_neg = len(y_train) - num_pos
            print(f"Classes: Neg (0)={num_neg}, Pos (1)={num_pos}")
            
            # Calculer pos_weight = neg / pos
            if num_pos > 0:
                pos_weight_val = num_neg / num_pos
                pos_weight = torch.tensor([pos_weight_val]).to(self.device)
                print(f"Poids pour la classe positive: {pos_weight_val:.4f}")
            else:
                pos_weight = None
                print("Attention: Pas de classe positive dans le train set!")
        except Exception as e:
            print(f"Erreur lors du calcul des poids: {e}")
            pos_weight = None
        
        # Fonction de perte et optimiseur
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCELoss()  # Fallback si pas de poids ou ancien modèle
            
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Historique
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    def train_epoch(self):
        """Entraînement pour une epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in tqdm(self.train_loader, desc="Training"):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            total_loss += loss.item()
            # Appliquer Sigmoid pour les métriques car le modèle sort des logits
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(batch_y.cpu().detach().numpy())
        
        # Calcul des métriques
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        return avg_loss, accuracy, f1
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.val_loader, desc="Validation"):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                # Appliquer Sigmoid
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Métriques
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        return avg_loss, accuracy, f1, all_preds, all_labels
    
    def early_stopping(self, val_loss):
        """Gérer l'early stopping"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            print(f"  ✓ Amélioration! Nouvelle best val loss: {val_loss:.4f}")
            return False
        else:
            self.patience_counter += 1
            print(f"  ✗ Pas d'amélioration ({self.patience_counter}/{self.patience})")
            return self.patience_counter >= self.patience
    
    def save_model(self, path=None):
        """Sauvegarder le modèle"""
        if path is None:
            path = self.config['paths']['model_save']
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Utiliser self.model.state_dict() si best_model_state est None
        model_state = self.best_model_state if self.best_model_state is not None else self.model.state_dict()
        
        torch.save({
            'model_state_dict': model_state,
            'config': self.config,
            'history': self.history
        }, path)
        
        print(f"✅ Modèle sauvegardé: {path}")
        print(f"   Taille: {os.path.getsize(path) / 1024:.2f} KB")
    
    def plot_training_history(self):
        """Visualiser l'historique d'entraînement"""
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Sauvegarder le plot
        save_path = self.config['paths']['results']
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/training_history.png')
        # plt.show()
    
    def train(self):
        """Boucle d'entraînement complète"""
        print(f"\n=== DÉBUT DE L'ENTRAÎNEMENT ===")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 40)
            
            # Entraînement
            train_loss, train_acc, train_f1 = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_f1, _, _ = self.validate()
            
            # Sauvegarder l'historique
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Afficher les résultats
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping à l'epoch {epoch+1}")
                break
        
        # Restaurer le meilleur modèle
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Visualiser l'historique
        self.plot_training_history()
        
        # Sauvegarder le modèle
        self.save_model()
        
        # Évaluation finale
        val_loss, val_acc, val_f1, _, _ = self.validate()
        print(f"\n=== RÉSULTATS FINAUX ===")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}")
        
        return self.history

def main():
    """Fonction principale d'entraînement"""
    trainer = Trainer()
    history = trainer.train()
    return history

if __name__ == "__main__":
    main()