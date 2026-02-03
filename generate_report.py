# generate_report.py
"""
================================================================================
PARTIE 5 : ÉVALUATION & ANALYSE DES RÉSULTATS
================================================================================
Objectif : Charger le modèle entraîné et le tester sur le jeu de Test (inédit).
Calcul des métriques (Accuracy, Recall, etc.) et génération des rapports.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc, roc_auc_score)
import pandas as pd
import os

print("=== RAPPORT COMPLET DU MODÈLE NF1 ===")

# 1. Charger le modèle et les données
print("\n1. Chargement des données et du modèle...")
model_path = 'models/nf1_classifier.pth'
if not os.path.exists(model_path):
    print(f"Erreur: Le modèle {model_path} n'existe pas.")
    exit(1)
checkpoint = torch.load(model_path, map_location='cpu')

test_data = torch.load('datasets/processed/test_data.pt')
X_test = test_data['X']
y_test = test_data['y'].numpy().flatten()

print(f"   Test set: {X_test.shape[0]} échantillons")

# 2. Recréer le modèle
# Importer la bonne classe
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.model.nf1_classifier import NF1Classifier

# Récupérer la config du checkpoint ou utiliser celle par défaut
if 'config' in checkpoint:
    config = checkpoint['config']
    input_size = config['model']['input_size']
    hidden_layers = config['model']['hidden_layers']
    dropout_rate = config['model']['dropout_rate']
    activation = config['model']['activation']
    print(f"   Config chargée depuis le checkpoint: { hidden_layers}")
else:
    # Fallback
    input_size = X_test.shape[1]
    hidden_layers = [64, 32, 16] 
    dropout_rate = 0.3
    activation = 'relu'
    print("   Utilisation de la config par défaut")

model = NF1Classifier(
    input_size=input_size,
    hidden_layers=hidden_layers,
    dropout_rate=dropout_rate, 
    activation=activation
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"   Modèle chargé: {model_path}")

# 3. Prédictions
print("\n2. Calcul des prédictions...")
with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.sigmoid(logits).numpy().flatten()
    predictions = (probabilities >= 0.5).astype(int)

# 4. Métriques détaillées
print("\n3. Métriques de performance:")
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc_score = roc_auc_score(y_test, probabilities)

print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   AUC-ROC:   {auc_score:.4f}")

# 5. Matrice de confusion détaillée
print("\n4. Matrice de confusion:")
cm = confusion_matrix(y_test, predictions)
print(f"\n   [TN FP]")
print(f"   [FN TP] = \n{cm}")

# Calculer les pourcentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

print(f"\n   Pourcentages par classe:")
print(f"   Sporadic correct: {cm_percent[0,0]:.1f}% ({cm[0,0]}/{cm[0,:].sum()})")
print(f"   Familial correct: {cm_percent[1,1]:.1f}% ({cm[1,1]}/{cm[1,:].sum()})")

# 6. Rapport de classification
print("\n5. Rapport de classification:")
print(classification_report(y_test, predictions, 
                           target_names=['Sporadic', 'Familial'],
                           digits=4))

# 7. Visualisations
print("\n6. Génération des visualisations...")
os.makedirs('results', exist_ok=True)

# Matrice de confusion visuelle
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sporadic', 'Familial'],
            yticklabels=['Sporadic', 'Familial'])
plt.title('Matrice de Confusion')
plt.ylabel('Vraie étiquette')
plt.xlabel('Étiquette prédite')

# Courbe ROC
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('results/confusion_matrix_roc.png', dpi=300, bbox_inches='tight')
# plt.show()

# 8. Distribution des probabilités
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(probabilities[y_test == 0], alpha=0.5, label='Sporadic', bins=20)
plt.hist(probabilities[y_test == 1], alpha=0.5, label='Familial', bins=20)
plt.xlabel('Probabilité prédite')
plt.ylabel('Fréquence')
plt.title('Distribution des Probabilités')
plt.legend()
plt.grid(True, alpha=0.3)

# Performance par seuil
thresholds = np.linspace(0, 1, 50)
accuracies = []
for thresh in thresholds:
    acc = accuracy_score(y_test, (probabilities >= thresh).astype(int))
    accuracies.append(acc)

plt.subplot(1, 2, 2)
plt.plot(thresholds, accuracies, 'b-')
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Seuil=0.5')
plt.xlabel('Seuil de décision')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Seuil de Décision')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/probability_distribution.png', dpi=300, bbox_inches='tight')
# plt.show()

# 9. Suggestions d'amélioration
print("\n" + "="*60)
print("ANALYSE ET SUGGESTIONS D'AMÉLIORATION")
print("="*60)

print("\n1. PERFORMANCE ACTUELLE:")
print(f"   • Accuracy: {accuracy:.1%} (légèrement > 50% aléatoire)")
print(f"   • AUC-ROC: {auc_score:.3f} (qualité discriminative modeste)")

print("\n2. PROBLÈMES IDENTIFIÉS:")
print(f"   • Faible recall pour 'Familial' ({recall:.1%})")
print(f"   • Déséquilibre de performance entre classes")
print(f"   • Le seuil de 0.5 n'est peut-être pas optimal")

print("\n3. SUGGESTIONS POUR AMÉLIORER:")

suggestions = [
    ("Rééquilibrage des classes", "Utiliser class_weight dans BCELoss ou oversampling"),
    ("Optimisation du seuil", f"Meilleur seuil trouvé: {thresholds[np.argmax(accuracies)]:.2f}"),
    ("Feature engineering", "Ajouter/créer des features plus discriminantes"),
    ("Architecture plus complexe", "Plus de couches/neurones, dropout différent"),
    ("Augmentation des données", "Techniques de data augmentation si applicable"),
    ("Validation croisée", "Meilleure estimation des performances"),
    ("Optimisation hyperparamètres", "Grid search pour learning rate, batch size, etc."),
    ("Essayer d'autres modèles", "XGBoost, SVM pour comparaison")
]

for i, (title, desc) in enumerate(suggestions, 1):
    print(f"   {i}. {title}: {desc}")

print("\n4. PROCHAINES ÉTAPES IMMÉDIATES:")
print("   • Exécuter: python main.py --mode test (avec le bon chemin)")
print("   • Analyser l'importance des features")
print("   • Ajuster le seuil de décision")
print("   • Essayer un rééquilibrage des classes")

# 10. Sauvegarder le rapport
report_text = f"""
RAPPORT D'ÉVALUATION - CLASSIFICATION NF1
{'='*50}

PERFORMANCES SUR LE TEST SET:
• Accuracy:  {accuracy:.4f}
• Precision: {precision:.4f}
• Recall:    {recall:.4f}
• F1-Score:  {f1:.4f}
• AUC-ROC:   {auc_score:.4f}

MATRICE DE CONFUSION:
{cm}

RAPPORT DE CLASSIFICATION:
{classification_report(y_test, predictions, target_names=['Sporadic', 'Familial'])}
"""

with open('results/evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✅ Rapport sauvegardé dans 'results/evaluation_report.txt'")
print(f"✅ Visualisations sauvegardées dans 'results/'")
print("\n" + "="*60)
print("PROJET TERMINÉ AVEC SUCCÈS!")
print("="*60)