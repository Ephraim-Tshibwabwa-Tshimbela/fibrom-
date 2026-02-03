# Rapport d'Évaluation des Résultats d'Entraînement et de Test - Classification NF1

## 1. Introduction
Ce rapport présente les performances du modèle de réseau de neurones développé pour la classification des cas de Neurofibromatose de type 1 (Familial vs Sporadique).

## 2. Métriques d'Entraînement
*(Insérer ici les graphiques de training_loss et validation_loss)*
- **Loss Finale (Train)** : [Valeur]
- **Loss Finale (Validation)** : [Valeur]
- **Accuracy Finale (Validation)** : 0.4444



L'analyse des courbes d'apprentissage (voir `results/training_history.png`) montre...

## 3. Métriques de Test
Le modèle a été évalué sur un jeu de test indépendant (15% du dataset).
- **Précision Globale (Accuracy)** : 0.4444
- **Matrice de Confusion** : (Voir `results/confusion_matrix_roc.png`)
    - Vrais Positifs (TP - Familial) : 10
    - Vrais Négatifs (TN - Sporadic) : 10
    - Faux Positifs (FP) : 14
    - Faux Négatifs (FN) : 11

## 4. Analyse des Caractéristiques Importantes
Les caractéristiques cliniques les plus influentes pour la prédiction sont :
1. [Caractéristique A]
2. [Caractéristique B]
3. [Caractéristique C]

## 5. Conclusion
suite au rééquilibrage des classes (poids ~1.2 pour la classe Familiale), nous observons :
- **Amélioration du Rappel** : Le modèle détecte mieux les cas familiaux (47.6% vs 38.1% auparavant).
- **Baisse de Précision** : Le nombre de Faux Positifs a augmenté (14 vs 7), ce qui fait chuter l'accuracy globale à 44.4%.
- **Bilan** : Le rééquilibrage a fonctionné pour "forcer" le modèle a s'intéresser à la classe minoritaire, mais au détriment de la précision générale. Cela suggère que les données sont difficiles à séparer linéairement avec ce modèle simple.
