# =============================================================================
# MODÈLE PMC (Perceptron Multi-Couches) - Pipeline complet
# =============================================================================
# Ce script applique l'algorithme PMC (MLP) sur deux jeux de données :
# 1. Australian Credit Approval (jeu imposé)
# 2. Wine Quality Red (jeu libre)
#
# Pour chaque jeu, on fait varier le nombre de neurones dans la couche cachée
# et on évalue les performances par validation croisée 5×5 (25 runs par config).
# =============================================================================

# -----------------------------------------------------------------------------
# ÉTAPE 1 : IMPORTS
# -----------------------------------------------------------------------------
# On importe toutes les librairies nécessaires au début du script

import numpy as np                                      # Pour les calculs (moyenne, écart-type)
import pandas as pd                                     # Pour charger et manipuler les données
from sklearn.preprocessing import normalize             # Pour normaliser les données
from sklearn.neural_network import MLPClassifier        # Le modèle PMC (MLP)
from sklearn.model_selection import RepeatedKFold       # Pour la validation croisée répétée
from sklearn.model_selection import cross_val_score     # Pour évaluer le modèle
import matplotlib
matplotlib.use('Agg')                                   # Backend non-interactif (pas de fenêtre)
import matplotlib.pyplot as plt                         # Pour afficher les graphiques
import warnings                                         # Pour ignorer les avertissements de convergence
warnings.filterwarnings('ignore')                       # Le MLP peut parfois ne pas converger parfaitement

# -----------------------------------------------------------------------------
# PARAMÈTRES GLOBAUX
# -----------------------------------------------------------------------------
# Ces paramètres seront utilisés pour les deux jeux de données

# Liste du nombre de neurones à tester dans la couche cachée
valeurs_neurones = [5, 10, 15, 20, 25, 30, 50]

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
# - Utilise des virgules comme séparateur
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
# La normalisation est importante pour le PMC car l'algorithme utilise
# la descente de gradient. Sans normalisation, l'apprentissage serait
# plus lent et moins stable.

print("[3] Normalisation des attributs...")
X_australian_normalise = normalize(X_australian)
print("    Normalisation terminée (norme L2)")

# -----------------------------------------------------------------------------
# ÉTAPE 2d : VALIDATION CROISÉE AVEC VARIATION DU NOMBRE DE NEURONES
# -----------------------------------------------------------------------------
# Pour chaque nombre de neurones, on entraîne et évalue le modèle PMC
# avec la validation croisée 5×5 (25 évaluations)

print("[4] Validation croisée pour différents nombres de neurones...")
print("    (Cela peut prendre un moment, le PMC est plus lent que le PPV...)\n")

# Listes pour stocker les résultats
resultats_australian_moyennes = []   # TCC moyen pour chaque config
resultats_australian_ecarts = []     # Écart-type pour chaque config

# Boucle sur toutes les valeurs de neurones à tester
for n_neurones in valeurs_neurones:
    # Création du modèle PMC avec une couche cachée de n_neurones
    # - hidden_layer_sizes=(n_neurones,) : une seule couche cachée avec n_neurones
    # - early_stopping=True : arrête l'entraînement si pas d'amélioration (évite sur-apprentissage)
    # - max_iter=1000 : nombre maximum d'itérations pour converger
    # - random_state=42 : pour la reproductibilité
    modele_pmc = MLPClassifier(
        hidden_layer_sizes=(n_neurones,),
        early_stopping=True,
        max_iter=1000,
        random_state=42
    )

    # Évaluation par validation croisée
    scores = cross_val_score(modele_pmc, X_australian_normalise, Y_australian,
                             cv=validation_croisee, scoring='accuracy')

    # Calcul de la moyenne et de l'écart-type des scores
    moyenne = np.mean(scores)
    ecart_type = np.std(scores)

    # Stockage des résultats
    resultats_australian_moyennes.append(moyenne)
    resultats_australian_ecarts.append(ecart_type)

    # Affichage de la progression
    print(f"    Neurones = {n_neurones:2d} → TCC = {moyenne * 100:.2f}%")

# -----------------------------------------------------------------------------
# ÉTAPE 2e : AFFICHAGE DU TABLEAU DE RÉSULTATS
# -----------------------------------------------------------------------------
# Trouver le meilleur nombre de neurones
meilleur_index_aus = np.argmax(resultats_australian_moyennes)
meilleur_neurones_aus = valeurs_neurones[meilleur_index_aus]
meilleur_tcc_aus = resultats_australian_moyennes[meilleur_index_aus]

print("\nRÉSULTATS - Australian Credit Approval")
print("-" * 45)
print(f"{'Neurones':<12} {'TCC moyen':<25}")
print("-" * 45)

for i, n in enumerate(valeurs_neurones):
    tcc = resultats_australian_moyennes[i] * 100       # Conversion en %
    ecart = resultats_australian_ecarts[i] * 100       # Conversion en %
    print(f"{n:<12} {tcc:.2f}% ± {ecart:.2f}%")

print("-" * 45)
print(f"Meilleur : {meilleur_neurones_aus} neurones (TCC = {meilleur_tcc_aus * 100:.2f}%)")

# -----------------------------------------------------------------------------
# ÉTAPE 2f : AFFICHAGE DU GRAPHIQUE
# -----------------------------------------------------------------------------
print("\n[5] Génération du graphique...")

plt.figure(figsize=(10, 6))

# Tracé de la courbe avec barres d'erreur (écart-type)
plt.errorbar(valeurs_neurones, resultats_australian_moyennes,
             yerr=resultats_australian_ecarts,
             marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
             label='TCC moyen', color='steelblue')

# Ligne verticale rouge pour indiquer le meilleur nombre de neurones
plt.axvline(x=meilleur_neurones_aus, color='red', linestyle='--', alpha=0.7,
            label=f'Meilleur = {meilleur_neurones_aus} neurones')

# Point rouge sur le meilleur pour le mettre en évidence
plt.scatter([meilleur_neurones_aus], [meilleur_tcc_aus], color='red', s=150, zorder=5)

plt.xlabel('Nombre de neurones (couche cachée)', fontsize=12)
plt.ylabel('TCC moyen (accuracy)', fontsize=12)
plt.title('PMC sur Australian Credit Approval\nTaux de classification correct en fonction du nombre de neurones', fontsize=14)
plt.xticks(valeurs_neurones)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig('resultats/pmc_australian.png', dpi=150)
print("    Graphique sauvegardé : resultats/pmc_australian.png")
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
# ÉTAPE 3d : VALIDATION CROISÉE AVEC VARIATION DU NOMBRE DE NEURONES
# -----------------------------------------------------------------------------
print("[4] Validation croisée pour différents nombres de neurones...")
print("    (Cela peut prendre un moment...)\n")

# Listes pour stocker les résultats
resultats_wine_moyennes = []
resultats_wine_ecarts = []

# Boucle sur toutes les valeurs de neurones à tester
for n_neurones in valeurs_neurones:
    # Création du modèle PMC
    modele_pmc = MLPClassifier(
        hidden_layer_sizes=(n_neurones,),
        early_stopping=True,
        max_iter=1000,
        random_state=42
    )

    # Évaluation par validation croisée
    scores = cross_val_score(modele_pmc, X_wine_normalise, Y_wine,
                             cv=validation_croisee, scoring='accuracy')

    # Calcul de la moyenne et de l'écart-type des scores
    moyenne = np.mean(scores)
    ecart_type = np.std(scores)

    # Stockage des résultats
    resultats_wine_moyennes.append(moyenne)
    resultats_wine_ecarts.append(ecart_type)

    # Affichage de la progression
    print(f"    Neurones = {n_neurones:2d} → TCC = {moyenne * 100:.2f}%")

# -----------------------------------------------------------------------------
# ÉTAPE 3e : AFFICHAGE DU TABLEAU DE RÉSULTATS
# -----------------------------------------------------------------------------
# Trouver le meilleur nombre de neurones
meilleur_index_wine = np.argmax(resultats_wine_moyennes)
meilleur_neurones_wine = valeurs_neurones[meilleur_index_wine]
meilleur_tcc_wine = resultats_wine_moyennes[meilleur_index_wine]

print("\nRÉSULTATS - Wine Quality Red")
print("-" * 45)
print(f"{'Neurones':<12} {'TCC moyen':<25}")
print("-" * 45)

for i, n in enumerate(valeurs_neurones):
    tcc = resultats_wine_moyennes[i] * 100             # Conversion en %
    ecart = resultats_wine_ecarts[i] * 100             # Conversion en %
    print(f"{n:<12} {tcc:.2f}% ± {ecart:.2f}%")

print("-" * 45)
print(f"Meilleur : {meilleur_neurones_wine} neurones (TCC = {meilleur_tcc_wine * 100:.2f}%)")

# -----------------------------------------------------------------------------
# ÉTAPE 3f : AFFICHAGE DU GRAPHIQUE
# -----------------------------------------------------------------------------
print("\n[5] Génération du graphique...")

plt.figure(figsize=(10, 6))

# Tracé de la courbe avec barres d'erreur (écart-type)
plt.errorbar(valeurs_neurones, resultats_wine_moyennes,
             yerr=resultats_wine_ecarts,
             marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
             label='TCC moyen', color='green')

# Ligne verticale rouge pour indiquer le meilleur nombre de neurones
plt.axvline(x=meilleur_neurones_wine, color='red', linestyle='--', alpha=0.7,
            label=f'Meilleur = {meilleur_neurones_wine} neurones')

# Point rouge sur le meilleur pour le mettre en évidence
plt.scatter([meilleur_neurones_wine], [meilleur_tcc_wine], color='red', s=150, zorder=5)

plt.xlabel('Nombre de neurones (couche cachée)', fontsize=12)
plt.ylabel('TCC moyen (accuracy)', fontsize=12)
plt.title('PMC sur Wine Quality Red\nTaux de classification correct en fonction du nombre de neurones', fontsize=14)
plt.xticks(valeurs_neurones)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Sauvegarde du graphique
plt.savefig('resultats/pmc_wine.png', dpi=150)
print("    Graphique sauvegardé : resultats/pmc_wine.png")
plt.show()


# =============================================================================
# FIN DU SCRIPT
# =============================================================================
print("\n" + "=" * 70)
print("SCRIPT TERMINÉ")
print("=" * 70)
print("\nRécapitulatif des fichiers générés :")
print("  - resultats/pmc_australian.png")
print("  - resultats/pmc_wine.png")
print("\nCes graphiques peuvent être utilisés dans le rapport.")
