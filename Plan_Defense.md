# Plan de Défense et Répartition du Projet

Ce document divise le projet en **5 parties distinctes** pour la soutenance. Chaque membre est responsable de maîtriser sa partie, ses fichiers, et de savoir faire la transition vers le membre suivant.

## Vue d'ensemble
Le projet est un pipeline de Deep Learning complet : Données → Prétraitement → Modèle → Entraînement → Évaluation.

---

## 👤 Partie 1 : Analyse Exploratoire & Compréhension des Données
**Objectif** : Présenter le problème médical (NF1), le dataset, et justifier les choix initiaux.

*   **Fichiers Clés** :
    *   `datasets/raw_data/` (le fichier Excel source)
    *   `Rapport d'Analyse du Dataset Neurof Tp DeepLearning.txt`
    *   `check_dimensions.py`
    *   `diagnostic.py`
*   **Concepts à Défendre** :
    *   La nature des données (cliniques, binaires/numériques).
    *   Le déséquilibre des classes (plus de Sporadique que de Familial) identifié dès le début.
    *   L'importance des dimensions (input_size = 20 caractéristiques).
*   **Transition** : "Maintenant que nous comprenons les données brutes, [Membre 2] va expliquer comment nous les avons rendues utilisables pour le réseau."

---

## 👤 Partie 2 : Pipeline de Prétraitement (ETL)
**Objectif** : Expliquer comment on passe du fichier Excel brut à des tenseurs PyTorch propres.

*   **Fichiers Clés** :
    *   `src/dataset/preprocessor.py`
    *   `src/dataset/loader.py` (Classe `NF1Dataset`)
    *   `datasets/processed/` (fichiers .pt)
*   **Concepts à Défendre** :
    *   Le nettoyage (gestion des valeurs manquantes, encodage).
    *   La normalisation (StandardScaler) : pourquoi c'est crucial pour les réseaux de neurones.
    *   La création des DataLoaders et le split Train/Val/Test (pour éviter la fuite de données).
*   **Transition** : "Les données étant prêtes et normalisées, [Membre 3] va vous présenter l'architecture neuronale que nous avons conçue."

---

## 👤 Partie 3 : Architecture du Modèle (Neural Network)
**Objectif** : Détailler le "cerveau" du projet et ses composants.

*   **Fichiers Clés** :
    *   `src/model/nf1_classifier.py`
    *   `config.yml` (section `model`)
*   **Concepts à Défendre** :
    *   Type de modèle : Perceptron Multicouche (MLP).
    *   Détails : Couches linéaires (`nn.Linear`), activation ReLU, Batch Normalization (pour la stabilité), Dropout (pour éviter l'overfitting).
    *   **Point important** : Expliquer pourquoi nous avons retiré la couche `Sigmoid` finale (pour utiliser `BCEWithLogitsLoss` et améliorer la stabilité numérique).
*   **Transition** : "Avec cette architecture définie, [Membre 4] va expliquer comment nous avons entraîné ce modèle pour qu'il apprenne."

---

## 👤 Partie 4 : Entraînement & Optimisation
**Objectif** : Montrer comment le modèle apprend et comment on a géré les difficultés (déséquilibre).

*   **Fichiers Clés** :
    *   `src/train.py` (Classe `Trainer`)
    *   `main.py` (Mode `train`)
*   **Concepts à Défendre** :
    *   Boucle d'entraînement : Forward pass, calcul de la Loss, Backward pass (rétropropagation).
    *   **Stratégie de Rééquilibrage** : L'utilisation de `pos_weight` dans `BCEWithLogitsLoss` pour donner plus d'importance aux cas rares (Familiaux).
    *   Early Stopping : Arrêter l'entraînement quand la validation ne s'améliore plus pour éviter le sur-apprentissage.
*   **Transition** : "Une fois le modèle entraîné, nous devons vérifier sa performance réelle. [Membre 5] va présenter les résultats."

---

## 👤 Partie 5 : Évaluation & Analyse des Résultats
**Objectif** : Être honnête sur les performances et proposer des améliorations.

*   **Fichiers Clés** :
    *   `generate_report.py`
    *   `Rapport_Evaluation.md`
    *   `results/` (Matrices de confusion, courbes)
*   **Concepts à Défendre** :
    *   Métriques utilisées : Accuracy, Précision, Rappel (Recall), F1-Score.
    *   Analyse critique : Nous avons sacrifié un peu d'Accuracy globale pour gagner en Rappel sur la classe malade (Familiale), ce qui est critique en médecine.
    *   Pistes d'amélioration futures (plus de données, architectures plus complexes).
*   **Conclusion** : Synthèse finale du projet.

---

## 🔄 Synchronisation
*   Assurez-vous que le **Membre 2** utilise bien les dimensions trouvées par le **Membre 1**.
*   Le **Membre 4** doit mentionner que la modification du modèle faite par le **Membre 3** (retrait Sigmoid) a obligé à changer la fonction de perte (Loss).
*   Le **Membre 5** doit faire le lien avec l'objectif du **Membre 1** (diagnostiquer correctement la NF1).
