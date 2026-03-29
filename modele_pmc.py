# Modèle PMC (Perceptron Multi-Couches)
# Lancer le programme : python modele_pmc.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paramètres : nombre de neurones à tester et validation croisée
valeurs_neurones = [5, 10, 15, 20, 25, 30, 50]
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

# Normalisation
X_australian_normalise = normalize(X_australian)

# Validation croisée pour chaque nombre de neurones
print("\nValidation croisée en cours\n")
resultats_australian_moyennes = []
resultats_australian_ecarts = []

for n in valeurs_neurones:
    modele = MLPClassifier(
        hidden_layer_sizes=(n,),
        activation='logistic',      # Sigmoïde
        early_stopping=True,        # Early stopping (sur-apprentissage)
        max_iter=1000,
        random_state=42
    )
    scores = cross_val_score(modele, X_australian_normalise, Y_australian,
                             cv=validation_croisee, scoring='accuracy')
    resultats_australian_moyennes.append(np.mean(scores))
    resultats_australian_ecarts.append(np.std(scores))
    print(f"  {n} neurones → {np.mean(scores)*100:.2f}%")

# Affichage des résultats
meilleur_idx = np.argmax(resultats_australian_moyennes)
meilleur_n_aus = valeurs_neurones[meilleur_idx]
meilleur_tcc_aus = resultats_australian_moyennes[meilleur_idx]

print("\nRésultats : Australian Credit Approval")
print("-" * 40)
for i, n in enumerate(valeurs_neurones):
    tcc = resultats_australian_moyennes[i] * 100
    ecart = resultats_australian_ecarts[i] * 100
    print(f"{n:2d} neurones  →  {tcc:.2f}% ± {ecart:.2f}%")
print("-" * 40)
print(f"Meilleur : {meilleur_n_aus} neurones (TCC = {meilleur_tcc_aus * 100:.2f}%)")

# Graphique
plt.figure(figsize=(10, 6))
plt.errorbar(valeurs_neurones, resultats_australian_moyennes, yerr=resultats_australian_ecarts,
             marker='o', capsize=5, linewidth=2, markersize=8, color='steelblue', label='TCC moyen')
plt.axvline(x=meilleur_n_aus, color='red', linestyle='--', alpha=0.7, label=f'Meilleur = {meilleur_n_aus} neurones')
plt.scatter([meilleur_n_aus], [meilleur_tcc_aus], color='red', s=150, zorder=5)
plt.xlabel('Nombre de neurones')
plt.ylabel('TCC moyen')
plt.title('PMC sur Australian Credit Approval')
plt.xticks(valeurs_neurones)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('resultats/pmc_australian.png', dpi=150)
print("\nGraphique : resultats/pmc_australian.png")


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

# Validation croisée pour chaque nombre de neurones
print("\nValidation croisée en cours\n")
resultats_wine_moyennes = []
resultats_wine_ecarts = []

for n in valeurs_neurones:
    modele = MLPClassifier(
        hidden_layer_sizes=(n,),
        activation='logistic',
        early_stopping=True,
        max_iter=1000,
        random_state=42
    )
    scores = cross_val_score(modele, X_wine_normalise, Y_wine,
                             cv=validation_croisee, scoring='accuracy')
    resultats_wine_moyennes.append(np.mean(scores))
    resultats_wine_ecarts.append(np.std(scores))
    print(f"  {n} neurones → {np.mean(scores)*100:.2f}%")

# Résultats et affichage
meilleur_idx = np.argmax(resultats_wine_moyennes)
meilleur_n_wine = valeurs_neurones[meilleur_idx]
meilleur_tcc_wine = resultats_wine_moyennes[meilleur_idx]

print("\nRésultats : Wine Quality Red")
print("-" * 40)
for i, n in enumerate(valeurs_neurones):
    tcc = resultats_wine_moyennes[i] * 100
    ecart = resultats_wine_ecarts[i] * 100
    print(f"{n:2d} neurones  →  {tcc:.2f}% ± {ecart:.2f}%")
print("-" * 40)
print(f"Meilleur : {meilleur_n_wine} neurones (TCC = {meilleur_tcc_wine * 100:.2f}%)")

# Graphique
plt.figure(figsize=(10, 6))
plt.errorbar(valeurs_neurones, resultats_wine_moyennes, yerr=resultats_wine_ecarts,
             marker='o', capsize=5, linewidth=2, markersize=8, color='green', label='TCC moyen')
plt.axvline(x=meilleur_n_wine, color='red', linestyle='--', alpha=0.7, label=f'Meilleur = {meilleur_n_wine} neurones')
plt.scatter([meilleur_n_wine], [meilleur_tcc_wine], color='red', s=150, zorder=5)
plt.xlabel('Nombre de neurones')
plt.ylabel('TCC moyen')
plt.title('PMC sur Wine Quality Red')
plt.xticks(valeurs_neurones)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('resultats/pmc_wine.png', dpi=150)
print("\nGraphique : resultats/pmc_wine.png")

print("\n" + "=" * 60)
print("PMC terminé")
print("=" * 60)
