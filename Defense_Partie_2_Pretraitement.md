# 🛡️ DÉFENSE PARTIE 2 : PRÉTRAITEMENT (ETL)

**Membre Responsable :** [Nom du Membre 2]
**Objectif :** Expliquer la transformation de la donnée brute en "nourriture" pour le réseau de neurones.

---

## 1. Le Pipeline de Nettoyage (`src/dataset/preprocessor.py`)
Le Deep Learning déteste 3 choses : le vide, le texte, et les grandes différences d'échelle.

### A. Gestion du Vide (NaN)
*   **Problème** : Le fichier Excel a des trous.
*   **Solution** : Nous remplaçons les valeurs manquantes par le **mode** (valeur la plus fréquente) de la colonne.
    ```python
    df[col] = df[col].fillna(df[col].mode()[0])
    ```

### B. Encodage (Texte -> Chiffres)
*   **Problème** : Le réseau ne lit que des maths, pas "Yes" ou "No".
*   **Solution** : `LabelEncoder`.
    *   "No" -> 0
    *   "Yes" -> 1
    *   "Sporadic" -> 0, "Familial" -> 1.

## 2. La Normalisation (Crucial !)
Certaines valeurs sont petites (0, 1) et d'autres grandes (Taille des taches). Sans normalisation, le modèle se focaliserait uniquement sur les grandes valeurs.
*   **Outil** : `StandardScaler` de Scikit-Learn.
    *   Moyenne -> 0
    *   Écart-type -> 1
*   **Résultat** : Toutes les features "pèsent" le même poids au départ.

## 3. Chargement PyTorch (`src/dataset/loader.py`)
Nous ne donnons pas tout le tableau d'un coup. Nous utilisons des **DataLoaders**.

### Le Split (Découpage)
*   **Train (70%)** : Pour apprendre.
*   **Val (15%)** : Pour vérifier en cours de route (Early stopping).
*   **Test (15%)** : Caché jusqu'à la fin (l'examen final).

### Le Batching
```python
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```
*   **Batch Size 32** : On apprend sur 32 patients à la fois (plus stable, plus rapide).
*   **Shuffle=True** : On mélange les cartes à chaque époque pour ne pas apprendre l'ordre des patients.

## 📋 Conclusion pour cette partie
"J'ai transformé un fichier Excel désordonné en matrices de tenseurs normalisées et découpées. Les données sont prêtes à entrer dans le réseau. Je laisse la parole pour l'architecture."
