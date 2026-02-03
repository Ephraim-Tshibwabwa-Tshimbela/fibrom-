import torch
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
import pandas as pd

from src.model.nf1_classifier import NF1Classifier
from src.dataset.loader import create_dataloaders

class Tester:
    """Classe pour l'évaluation du modèle"""
    
    def __init__(self, model_path=None, config_path="config.yml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Charger le modèle
        if model_path is None:
            model_path = self.config['paths']['model_save']
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Créer les dataloaders
        self.train_loader, self.val_loader, self.test_loader, input_size = \
            create_dataloaders(config_path)
        
        # Initialiser le modèle avec la configuration sauvegardée
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
            self.model = NF1Classifier(
                input_size=model_config['input_size'],
                hidden_layers=model_config['hidden_layers'],
                dropout_rate=model_config['dropout_rate'],
                activation=model_config['activation']
            )
        else:
            # Fallback: utiliser la config actuelle
            self.model = NF1Classifier(
                input_size=self.config['model']['input_size'],
                hidden_layers=self.config['model']['hidden_layers'],
                dropout_rate=self.config['model']['dropout_rate'],
                activation=self.config['model']['activation']
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Modèle chargé depuis: {model_path}")
        print(f"Taille d'entrée du modèle: {self.model.config['input_size']}")
        print(f"Évaluation sur {self.device}")
    
    def evaluate(self, loader, name="Test"):
        """Évaluation complète sur un loader"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                preds = (outputs >= 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        # Convertir en arrays numpy
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        all_probs = np.array(all_probs).flatten()
        
        # Calculer les métriques
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'auc': roc_auc_score(all_labels, all_probs)
        }
        
        # Matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        
        # Rapport de classification
        report = classification_report(
            all_labels, all_preds,
            target_names=['Sporadic', 'Familial'],
            output_dict=True
        )
        
        print(f"\n{'='*60}")
        print(f"ÉVALUATION - {name}")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"AUC-ROC:   {metrics['auc']:.4f}")
        
        return metrics, cm, report, all_probs, all_labels
    
    def plot_confusion_matrix(self, cm, title="Matrice de Confusion"):
        """Visualiser la matrice de confusion"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Sporadic', 'Familial'],
                   yticklabels=['Sporadic', 'Familial'])
        plt.title(title)
        plt.ylabel('Vraies étiquettes')
        plt.xlabel('Étiquettes prédites')
        
        # Sauvegarder
        save_path = self.config['paths']['results']
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_probs, title="Courbe ROC"):
        """Visualiser la courbe ROC"""
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Sauvegarder
        save_path = self.config['paths']['results']
        plt.savefig(f'{save_path}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, feature_names=None):
        """Analyser l'importance des features"""
        # Charger quelques échantillons
        batch_x, _ = next(iter(self.train_loader))
        batch_x = batch_x.to(self.device)
        
        # Calculer l'importance
        importance = self.model.get_feature_importance(batch_x)
        
        # Noms des features par défaut
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        
        # Créer un DataFrame pour visualisation
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Visualiser
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
        plt.xlabel('Importance')
        plt.title('Top 15 des Features les plus Importantes')
        plt.gca().invert_yaxis()
        
        # Sauvegarder
        save_path = self.config['paths']['results']
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def generate_report(self):
        """Générer un rapport complet"""
        print("GÉNÉRATION DU RAPPORT D'ÉVALUATION")
        print("="*60)
        
        # Évaluation sur les trois sets
        train_metrics, train_cm, train_report, _, _ = self.evaluate(self.train_loader, "Train")
        val_metrics, val_cm, val_report, _, _ = self.evaluate(self.val_loader, "Validation")
        test_metrics, test_cm, test_report, test_probs, test_labels = self.evaluate(self.test_loader, "Test")
        
        # Visualisations
        self.plot_confusion_matrix(test_cm, "Matrice de Confusion - Test")
        self.plot_roc_curve(test_labels, test_probs, "Courbe ROC - Test")
        
        # Analyse d'importance des features
        # Charger les noms des features depuis les données brutes
        try:
            import pandas as pd
            df = pd.read_excel(self.config['data']['data_path'])
            feature_names = list(df.columns)
            feature_names.remove('Case Type')  # Retirer la target
            # Ajouter les features d'engineering
            feature_names.extend(['Parent_Age_Diff', 'Diagnosis_Ratio'])
            
            importance_df = self.analyze_feature_importance(feature_names)
            print("\nTop 10 des Features les plus Importantes:")
            print(importance_df.head(10))
        except:
            print("\nImpossible de charger les noms des features")
            importance_df = None
        
        # Créer un résumé des résultats
        summary = pd.DataFrame({
            'Train': train_metrics,
            'Validation': val_metrics,
            'Test': test_metrics
        }).T
        
        print("\n" + "="*60)
        print("RÉSUMÉ DES RÉSULTATS")
        print("="*60)
        print(summary)
        
        # Sauvegarder le rapport
        save_path = self.config['paths']['results']
        os.makedirs(save_path, exist_ok=True)
        
        summary.to_csv(f'{save_path}/results_summary.csv')
        
        # Rapport détaillé
        with open(f'{save_path}/detailed_report.txt', 'w') as f:
            f.write("RAPPORT DÉTAILLÉ D'ÉVALUATION\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. RÉSUMÉ DES MÉTRIQUES\n")
            f.write(summary.to_string() + "\n\n")
            
            f.write("2. RAPPORT DE CLASSIFICATION - TEST\n")
            f.write(classification_report(test_labels, (test_probs >= 0.5).astype(int),
                                       target_names=['Sporadic', 'Familial']))
            
            f.write("\n3. MATRICE DE CONFUSION - TEST\n")
            f.write(str(test_cm) + "\n")
        
        print(f"\nRapport sauvegardé dans: {save_path}")
        
        return summary, importance_df

def main():
    """Fonction principale de test"""
    tester = Tester()
    summary, importance_df = tester.generate_report()
    
    return summary, importance_df

if __name__ == "__main__":
    main()