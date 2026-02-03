# 🛡️ DÉFENSE PARTIE 1 : ANALYSE DES DONNÉES

**Membre Responsable :** [Nom du Membre 1]
**Objectif :** Poser les bases. Montrer que l'on comprend le problème médical et les données avant de coder.

---

## 1. Le Contexte Médical (En Bref)
*   **Sujet** : La Neurofibromatose (NF1).
*   **But** : Prédire si un cas est **Familial** (Héréditaire) ou **Sporadique** (Mutation spontanée).
*   **Pourquoi l'IA ?** : Aider au diagnostic précoce à partir de symptômes cliniques.

## 2. Exploration des Données
Nous sommes partis d'un fichier Excel brut (`dataset-uci.xlsx`).

### Structure du Dataset
*   **31 colonnes au départ** (beaucoup de bruit).
*   **Cible (Target)** : La colonne `Genetic Disorder` (Sporadic vs Familial).
*   **Dimensions Finales** : Nous avons identifié **20 caractéristiques (features)** pertinentes après nettoyage.

### Le Problème Majeur : Le Déséquilibre
Le dataset n'est pas "propre" naturellement.
> *"En analysant la colonne cible, nous avons vu une répartition inégale."*

| Classe | Nombre (Train set observé) |
| :--- | :--- |
| **Sporadic (Négatif)** | ~113 cas |
| **Familial (Positif)** | ~94 cas |

Cela crée un biais : le modèle risque de favoriser la classe majoritaire (Sporadic). C'est un point clé que nous avons signalé pour la partie Entraînement.

## 3. Analyse du Code (`check_dimensions.py` / `diagnostic.py`)
Nous avons créé des scripts pour automatiser cette analyse.

```python
# Extrait de diagnostic.py
class_counts = df[target_col].value_counts()
print(f"Distribution :\n{class_counts}")
```

**Ce que cela nous a appris :**
1.  Il y a des valeurs manquantes (NaN) qu'il faut gérer.
2.  Certaines colonnes sont du texte ("Yes"/"No") -> à convertir.
3.  L'échelle des valeurs varie (âge vs taille des taches) -> à normaliser.

## 📋 Conclusion pour cette partie
"Mesdames, Messieurs les jurés, l'analyse a révélé un jeu de données complexe et déséquilibré. J'ai identifié les 20 marqueurs clés et préparé le terrain pour que mon collègue puisse nettoyer ces données. Je passe la parole à la Partie 2."
