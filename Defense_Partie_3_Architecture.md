# 🛡️ DÉFENSE PARTIE 3 : ARCHITECTURE DU MODÈLE

**Membre Responsable :** [Nom du Membre 3]
**Objectif :** Présenter le "Cerveau" artificiel.

---

## 1. Vue d'ensemble : Le Perceptron Multicouche (MLP)
Nous avons choisi une architecture simple mais robuste pour ces données tabulaires.

```mermaid
graph LR
    Input[Entrée (20 Features)] --> Hidden1[Cachée 1 (64 Neurones)]
    Hidden1 --> Hidden2[Cachée 2 (32 Neurones)]
    Hidden2 --> Hidden3[Cachée 3 (16 Neurones)]
    Hidden3 --> Output[Sortie (1 Logit)]
```

## 2. Les Composants (`src/model/nf1_classifier.py`)

### A. Linear (Fully Connected)
Ce sont les neurones qui font les calculs ($y = wx + b$).
*   Couche 1 : 20 antécédents -> 64 neurones.
*   Couche 2 : 64 -> 32.
*   Couche 3 : 32 -> 16.

### B. Activation (ReLU)
Sans activation, le réseau ne serait qu'une régression linéaire géante. **ReLU** (Rectified Linear Unit) permet d'apprendre des relations non-linéaires complexes.

### C. Stabilisation & Généralisation
*   **Batch Normalization** : Recentre les données entre chaque couche.
    *   *Pourquoi ?* Ça accélère l'apprentissage et évite que les neurones ne "meurent".
*   **Dropout (0.3)** : On éteint aléatoirement 30% des neurones à chaque passage.
    *   *Pourquoi ?* Pour forcer le réseau à être robuste et ne pas apprendre par cœur (Overfitting).

## 3. Le Point Critique : "Logits" vs "Probabilités"
Initialement, nous avions une `Sigmoid` à la fin pour sortir une probabilité (0 à 1).
**Nous l'avons retirée.**

> *"Pourquoi ce changement ?"*
> Pour utiliser la fonction de perte `BCEWithLogitsLoss`. Elle combine la Sigmoid et la Loss en une seule opération mathématique plus stable numériquement. Le modèle sort donc un "Logit" (nombre brut, ex: 2.5 ou -1.2) que nous transformons en probabilité seulement au moment de la prédiction.

## 📋 Conclusion pour cette partie
"Nous avons construit un réseau profond mais contrôlé (Dropout/BatchNorm) et optimisé pour la stabilité numérique. Il est prêt à apprendre."
