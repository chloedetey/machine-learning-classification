# Modèle PPV (K-Plus Proches Voisins)
# Lancer le programme : python modele_ppv.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paramètres : nb de voisins à tester et validation croisée
valeurs_k = [1, 3, 5, 7, 9, 11, 15, 21]
validation_croisee = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)


# Dataset 1
print("=" * 60)
print("DATASET 1 : AUSTRALIAN CREDIT APPROVAL")
print("=" * 60)

# Chargement des données
donnees_australian = pd.read_csv("datasets/australian.data", sep=r',\s*', header=None, engine='python')
print(f"Dimensions : {donnees_australian.shape[0]} exemples, {donnees_australian.shape[1]} colonnes")

# Séparation X (attributs) et Y (classe)
X_australian = donnees_australian.iloc[:, :-1].values
Y_australian = donnees_australian.iloc[:, -1].values

# Normalisation pour le calcul de distances
X_australian_normalise = normalize(X_australian)

# Validation croisée pour chaque valeur de K
print("\nValidation croisée en cours\n")
resultats_australian_moyennes = []
resultats_australian_ecarts = []

for k in valeurs_k:
    modele = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(modele, X_australian_normalise, Y_australian,
                             cv=validation_croisee, scoring='accuracy')
    resultats_australian_moyennes.append(np.mean(scores))
    resultats_australian_ecarts.append(np.std(scores))

# Résultats et affichage
meilleur_idx = np.argmax(resultats_australian_moyennes)
meilleur_k_aus = valeurs_k[meilleur_idx]
meilleur_tcc_aus = resultats_australian_moyennes[meilleur_idx]

print("Résultats : Australian Credit Approval")
print("-" * 40)
for i, k in enumerate(valeurs_k):
    tcc = resultats_australian_moyennes[i] * 100
    ecart = resultats_australian_ecarts[i] * 100
    print(f"K = {k:2d}  →  {tcc:.2f}% ± {ecart:.2f}%")
print("-" * 40)
print(f"Meilleur : K = {meilleur_k_aus} (TCC = {meilleur_tcc_aus * 100:.2f}%)")

# Graphique
plt.figure(figsize=(10, 6))
plt.errorbar(valeurs_k, resultats_australian_moyennes, yerr=resultats_australian_ecarts,
             marker='o', capsize=5, linewidth=2, markersize=8, color='steelblue', label='TCC moyen')
plt.axvline(x=meilleur_k_aus, color='red', linestyle='--', alpha=0.7, label=f'Meilleur K = {meilleur_k_aus}')
plt.scatter([meilleur_k_aus], [meilleur_tcc_aus], color='red', s=150, zorder=5)
plt.xlabel('K (nombre de voisins)')
plt.ylabel('TCC moyen')
plt.title('PPV sur Australian Credit Approval')
plt.xticks(valeurs_k)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('resultats/ppv_australian.png', dpi=150)
print("\nGraphique : resultats/ppv_australian.png")


# Dataset 2
print("\n" + "=" * 60)
print("DATASET 2 : WINE QUALITY RED")
print("=" * 60)

# Chargement des données
donnees_wine = pd.read_csv("datasets/winequality-red.csv", sep=";")
print(f"Dimensions : {donnees_wine.shape[0]} exemples, {donnees_wine.shape[1]} colonnes")

# Séparation X et Y (transformation en classes binaires : >= 6 = bon vin)
X_wine = donnees_wine.drop("quality", axis=1).values
Y_wine = (donnees_wine["quality"].values >= 6).astype(int)
print(f"Classes : {np.sum(Y_wine == 0)} ordinaires, {np.sum(Y_wine == 1)} bons")

# Normalisation
X_wine_normalise = normalize(X_wine)

# Validation croisée pour chaque valeur de K
print("\nValidation croisée en cours\n")
resultats_wine_moyennes = []
resultats_wine_ecarts = []

for k in valeurs_k:
    modele = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(modele, X_wine_normalise, Y_wine,
                             cv=validation_croisee, scoring='accuracy')
    resultats_wine_moyennes.append(np.mean(scores))
    resultats_wine_ecarts.append(np.std(scores))

# Résultats et affichage
meilleur_idx = np.argmax(resultats_wine_moyennes)
meilleur_k_wine = valeurs_k[meilleur_idx]
meilleur_tcc_wine = resultats_wine_moyennes[meilleur_idx]

print("Résultats : Wine Quality Red")
print("-" * 40)
for i, k in enumerate(valeurs_k):
    tcc = resultats_wine_moyennes[i] * 100
    ecart = resultats_wine_ecarts[i] * 100
    print(f"K = {k:2d}  →  {tcc:.2f}% ± {ecart:.2f}%")
print("-" * 40)
print(f"Meilleur : K = {meilleur_k_wine} (TCC = {meilleur_tcc_wine * 100:.2f}%)")

# Graphique
plt.figure(figsize=(10, 6))
plt.errorbar(valeurs_k, resultats_wine_moyennes, yerr=resultats_wine_ecarts,
             marker='o', capsize=5, linewidth=2, markersize=8, color='green', label='TCC moyen')
plt.axvline(x=meilleur_k_wine, color='red', linestyle='--', alpha=0.7, label=f'Meilleur K = {meilleur_k_wine}')
plt.scatter([meilleur_k_wine], [meilleur_tcc_wine], color='red', s=150, zorder=5)
plt.xlabel('K (nombre de voisins)')
plt.ylabel('TCC moyen')
plt.title('PPV sur Wine Quality Red')
plt.xticks(valeurs_k)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('resultats/ppv_wine.png', dpi=150)
print("\nGraphique : resultats/ppv_wine.png")

print("\n" + "=" * 60)
print("PPV terminé")
print("=" * 60)
