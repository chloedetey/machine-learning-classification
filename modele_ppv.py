# =============================================================================
# MODÈLE PPV (K-Plus Proches Voisins) - Pipeline complet
# =============================================================================
# Ce script applique l'algorithme PPV (KNN) sur deux jeux de données :
# 1. Australian Credit Approval (jeu imposé)
# 2. Wine Quality Red (jeu libre)
#
# Pour chaque jeu, on fait varier le nombre de voisins K et on évalue
# les performances par validation croisée 5×5 (25 runs par configuration).
# =============================================================================

# -----------------------------------------------------------------------------
# ÉTAPE 1 : IMPORTS
# -----------------------------------------------------------------------------
# On importe toutes les librairies nécessaires au début du script

import numpy as np                                      # Pour les calculs (moyenne, écart-type)
import pandas as pd                                     # Pour charger et manipuler les données
from sklearn.preprocessing import normalize             # Pour normaliser les données
from sklearn.neighbors import KNeighborsClassifier      # Le modèle PPV (KNN)
from sklearn.model_selection import RepeatedKFold       # Pour la validation croisée répétée
from sklearn.model_selection import cross_val_score     # Pour évaluer le modèle
import matplotlib
matplotlib.use('Agg')                                   # Backend non-interactif (pas de fenêtre)
import matplotlib.pyplot as plt                         # Pour afficher les graphiques

# -----------------------------------------------------------------------------
# PARAMÈTRES GLOBAUX
# -----------------------------------------------------------------------------
# Ces paramètres seront utilisés pour les deux jeux de données

# Liste des valeurs de K (nombre de voisins) à tester
valeurs_k = [1, 3, 5, 7, 9, 11, 15, 21]

# Configuration de la validation croisée :
# - n_splits=5 : on divise les données en 5 parties (folds)
# - n_repeats=5 : on répète ce processus 5 fois avec des mélanges différents
# - Cela donne 5 × 5 = 25 évaluations par configuration
# - random_state=42 : pour avoir des résultats reproductibles
validation_croisee = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)


# =============================================================================
# DATASET 1 : AUSTRALIAN CREDIT APPROVAL
# =============================================================================
print("=" * 70)
print("DATASET 1 : AUSTRALIAN CREDIT APPROVAL")
print("=" * 70)

# -----------------------------------------------------------------------------
# ÉTAPE 2a : CHARGEMENT DES DONNÉES AUSTRALIAN
# -----------------------------------------------------------------------------
# Le fichier australian.data :
# - N'a pas d'en-tête (header=None)
# - Utilise des virgules comme séparateur (sep=',')
# - Contient 690 exemples et 15 colonnes (14 attributs + 1 classe)

print("\n[1] Chargement des données...")
donnees_australian = pd.read_csv("datasets/australian.data", sep=r',\s*', header=None, engine='python')
print(f"    Dimensions : {donnees_australian.shape[0]} exemples, {donnees_australian.shape[1]} colonnes")

# -----------------------------------------------------------------------------
# ÉTAPE 2b : SÉPARATION DES ATTRIBUTS (X) ET DE LA CLASSE (Y)
# -----------------------------------------------------------------------------
# - X = toutes les colonnes sauf la dernière (les 14 attributs)
# - Y = la dernière colonne (la classe : 0 ou 1)

print("[2] Séparation des attributs (X) et de la classe (Y)...")
X_australian = donnees_australian.iloc[:, :-1].values   # Toutes les colonnes sauf la dernière
Y_australian = donnees_australian.iloc[:, -1].values    # La dernière colonne uniquement
print(f"    X : {X_australian.shape[0]} exemples, {X_australian.shape[1]} attributs")
print(f"    Y : {len(Y_australian)} étiquettes de classe")

# -----------------------------------------------------------------------------
# ÉTAPE 2c : NORMALISATION DES DONNÉES
# -----------------------------------------------------------------------------
# La normalisation est importante pour le PPV car l'algorithme utilise
# la distance euclidienne. Sans normalisation, les attributs avec de grandes
# valeurs domineraient le calcul de distance.

print("[3] Normalisation des attributs...")
X_australian_normalise = normalize(X_australian)
print("    Normalisation terminée (norme L2)")

# -----------------------------------------------------------------------------
# ÉTAPE 2d : VALIDATION CROISÉE AVEC VARIATION DE K
# -----------------------------------------------------------------------------
# Pour chaque valeur de K, on entraîne et évalue le modèle PPV
# avec la validation croisée 5×5 (25 évaluations)

print("[4] Validation croisée pour différentes valeurs de K...")
print("    (Cela peut prendre quelques secondes...)\n")

# Listes pour stocker les résultats
resultats_australian_moyennes = []   # TCC moyen pour chaque K
resultats_australian_ecarts = []     # Écart-type pour chaque K

# Boucle sur toutes les valeurs de K à tester
for k in valeurs_k:
    # Création du modèle PPV avec K voisins
    modele_ppv = KNeighborsClassifier(n_neighbors=k)

    # Évaluation par validation croisée
    # cross_val_score retourne les scores (accuracy) pour chaque fold
    scores = cross_val_score(modele_ppv, X_australian_normalise, Y_australian,
                             cv=validation_croisee, scoring='accuracy')

    # Calcul de la moyenne et de l'écart-type des scores
    moyenne = np.mean(scores)
    ecart_type = np.std(scores)

    # Stockage des résultats
    resultats_australian_moyennes.append(moyenne)
    resultats_australian_ecarts.append(ecart_type)

# -----------------------------------------------------------------------------
# ÉTAPE 2e : AFFICHAGE DU TABLEAU DE RÉSULTATS
# -----------------------------------------------------------------------------
# Trouver le meilleur K (celui avec le TCC moyen le plus élevé)
meilleur_index_aus = np.argmax(resultats_australian_moyennes)
meilleur_k_aus = valeurs_k[meilleur_index_aus]
meilleur_tcc_aus = resultats_australian_moyennes[meilleur_index_aus]

print("RÉSULTATS - Australian Credit Approval")
print("-" * 40)
print(f"{'K':<8} {'TCC moyen':<25}")
print("-" * 40)

for i, k in enumerate(valeurs_k):
    tcc = resultats_australian_moyennes[i] * 100       # Conversion en %
    ecart = resultats_australian_ecarts[i] * 100       # Conversion en %
    print(f"{k:<8} {tcc:.2f}% ± {ecart:.2f}%")

print("-" * 40)
print(f"Meilleur K : {meilleur_k_aus} (TCC = {meilleur_tcc_aus * 100:.2f}%)")

# -----------------------------------------------------------------------------
# ÉTAPE 2f : AFFICHAGE DU GRAPHIQUE
# -----------------------------------------------------------------------------
print("\n[5] Génération du graphique...")

plt.figure(figsize=(10, 6))

# Tracé de la courbe avec barres d'erreur (écart-type)
plt.errorbar(valeurs_k, resultats_australian_moyennes,
             yerr=resultats_australian_ecarts,
             marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
             label='TCC moyen', color='steelblue')

# Ligne verticale rouge pour indiquer le meilleur K
plt.axvline(x=meilleur_k_aus, color='red', linestyle='--', alpha=0.7,
            label=f'Meilleur K = {meilleur_k_aus}')

# Point rouge sur le meilleur K pour le mettre en évidence
plt.scatter([meilleur_k_aus], [meilleur_tcc_aus], color='red', s=150, zorder=5)

plt.xlabel('K (nombre de voisins)', fontsize=12)
plt.ylabel('TCC moyen (accuracy)', fontsize=12)
plt.title('PPV sur Australian Credit Approval\nTaux de classification correct en fonction de K', fontsize=14)
plt.xticks(valeurs_k)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig('resultats/ppv_australian.png', dpi=150)
print("    Graphique sauvegardé : resultats/ppv_australian.png")
plt.show()


# =============================================================================
# DATASET 2 : WINE QUALITY RED
# =============================================================================
print("\n" + "=" * 70)
print("DATASET 2 : WINE QUALITY RED")
print("=" * 70)

# -----------------------------------------------------------------------------
# ÉTAPE 3a : CHARGEMENT DES DONNÉES WINE
# -----------------------------------------------------------------------------
# Le fichier winequality-red.csv :
# - A un en-tête (première ligne = noms des colonnes)
# - Utilise le point-virgule comme séparateur (sep=";")
# - Contient 1599 exemples et 12 colonnes (11 attributs + 1 colonne quality)

print("\n[1] Chargement des données...")
donnees_wine = pd.read_csv("datasets/winequality-red.csv", sep=";")
print(f"    Dimensions : {donnees_wine.shape[0]} exemples, {donnees_wine.shape[1]} colonnes")

# -----------------------------------------------------------------------------
# ÉTAPE 3b : SÉPARATION ET TRANSFORMATION DE LA CLASSE
# -----------------------------------------------------------------------------
# - X = toutes les colonnes sauf "quality" (les 11 attributs)
# - Y = la colonne "quality" transformée en 2 classes :
#       * quality >= 6 → classe 1 ("bon vin")
#       * quality < 6  → classe 0 ("vin ordinaire")

print("[2] Séparation des attributs (X) et transformation de la classe (Y)...")

# Extraction des attributs (toutes les colonnes sauf "quality")
X_wine = donnees_wine.drop("quality", axis=1).values

# Extraction et transformation de la classe en binaire (0 ou 1)
# La colonne "quality" contient des notes de 3 à 8
# On la transforme : >= 6 devient 1, < 6 devient 0
Y_wine_original = donnees_wine["quality"].values
Y_wine = (Y_wine_original >= 6).astype(int)

print(f"    X : {X_wine.shape[0]} exemples, {X_wine.shape[1]} attributs")
print(f"    Y : {len(Y_wine)} étiquettes de classe")
print(f"    Distribution des classes : {np.sum(Y_wine == 0)} vins ordinaires (0), {np.sum(Y_wine == 1)} bons vins (1)")

# -----------------------------------------------------------------------------
# ÉTAPE 3c : NORMALISATION DES DONNÉES
# -----------------------------------------------------------------------------
print("[3] Normalisation des attributs...")
X_wine_normalise = normalize(X_wine)
print("    Normalisation terminée (norme L2)")

# -----------------------------------------------------------------------------
# ÉTAPE 3d : VALIDATION CROISÉE AVEC VARIATION DE K
# -----------------------------------------------------------------------------
print("[4] Validation croisée pour différentes valeurs de K...")
print("    (Cela peut prendre quelques secondes...)\n")

# Listes pour stocker les résultats
resultats_wine_moyennes = []
resultats_wine_ecarts = []

# Boucle sur toutes les valeurs de K à tester
for k in valeurs_k:
    # Création du modèle PPV avec K voisins
    modele_ppv = KNeighborsClassifier(n_neighbors=k)

    # Évaluation par validation croisée
    scores = cross_val_score(modele_ppv, X_wine_normalise, Y_wine,
                             cv=validation_croisee, scoring='accuracy')

    # Calcul de la moyenne et de l'écart-type des scores
    moyenne = np.mean(scores)
    ecart_type = np.std(scores)

    # Stockage des résultats
    resultats_wine_moyennes.append(moyenne)
    resultats_wine_ecarts.append(ecart_type)

# -----------------------------------------------------------------------------
# ÉTAPE 3e : AFFICHAGE DU TABLEAU DE RÉSULTATS
# -----------------------------------------------------------------------------
# Trouver le meilleur K
meilleur_index_wine = np.argmax(resultats_wine_moyennes)
meilleur_k_wine = valeurs_k[meilleur_index_wine]
meilleur_tcc_wine = resultats_wine_moyennes[meilleur_index_wine]

print("RÉSULTATS - Wine Quality Red")
print("-" * 40)
print(f"{'K':<8} {'TCC moyen':<25}")
print("-" * 40)

for i, k in enumerate(valeurs_k):
    tcc = resultats_wine_moyennes[i] * 100             # Conversion en %
    ecart = resultats_wine_ecarts[i] * 100             # Conversion en %
    print(f"{k:<8} {tcc:.2f}% ± {ecart:.2f}%")

print("-" * 40)
print(f"Meilleur K : {meilleur_k_wine} (TCC = {meilleur_tcc_wine * 100:.2f}%)")

# -----------------------------------------------------------------------------
# ÉTAPE 3f : AFFICHAGE DU GRAPHIQUE
# -----------------------------------------------------------------------------
print("\n[5] Génération du graphique...")

plt.figure(figsize=(10, 6))

# Tracé de la courbe avec barres d'erreur (écart-type)
plt.errorbar(valeurs_k, resultats_wine_moyennes,
             yerr=resultats_wine_ecarts,
             marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
             label='TCC moyen', color='green')

# Ligne verticale rouge pour indiquer le meilleur K
plt.axvline(x=meilleur_k_wine, color='red', linestyle='--', alpha=0.7,
            label=f'Meilleur K = {meilleur_k_wine}')

# Point rouge sur le meilleur K pour le mettre en évidence
plt.scatter([meilleur_k_wine], [meilleur_tcc_wine], color='red', s=150, zorder=5)

plt.xlabel('K (nombre de voisins)', fontsize=12)
plt.ylabel('TCC moyen (accuracy)', fontsize=12)
plt.title('PPV sur Wine Quality Red\nTaux de classification correct en fonction de K', fontsize=14)
plt.xticks(valeurs_k)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig('resultats/ppv_wine.png', dpi=150)
print("    Graphique sauvegardé : resultats/ppv_wine.png")
plt.show()


# =============================================================================
# FIN DU SCRIPT
# =============================================================================
print("\n" + "=" * 70)
print("SCRIPT TERMINÉ")
print("=" * 70)
print("\nRécapitulatif des fichiers générés :")
print("  - resultats/ppv_australian.png")
print("  - resultats/ppv_wine.png")
