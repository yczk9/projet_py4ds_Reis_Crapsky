import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Import des données
from import_fao import get_datas_fao

liste_pays = [202, 185, 138, 100, 351, 21]  # Codes des pays BRICSAM
df = get_datas_fao(liste_pays)

# Création d'un pivot pour avoir les indicateurs en colonnes
df_pivot = df.pivot_table(
    index=['Pays', 'Année'],
    columns='Indicateur',
    values='Valeur'
).reset_index()

# Renommage des colonnes pour plus de lisibilité
df_pivot.columns.name = None

# Inversion de la prévalence de la sous-alimentation (21004)
# Car elle évolue dans le sens opposé aux autres indicateurs
# Une valeur élevée de sous-alimentation = mauvaise sécurité alimentaire
# Donc on prend l'opposé pour avoir une cohérence directionnelle
df_pivot['prévalence de la sous alimentation'] = -df_pivot['prévalence de la sous alimentation']

# Variables retenues pour l'ACP (nos 4 indicateurs)
variables_acp = [
    'suffisance des apports énergétiques alimentaires moyens',
    'disponibilité alimentaire par habitant',
    'disponibilité protéiques moyenne',
    'prévalence de la sous alimentation'  # Maintenant inversée
]

# Extraction des données pour l'ACP (suppression des valeurs manquantes)
df_acp = df_pivot[['Pays', 'Année'] + variables_acp].dropna()

# Séparation des identifiants et des variables quantitatives
X = df_acp[variables_acp]
identifiants = df_acp[['Pays', 'Année']]

print(f"Nombre d'observations pour l'ACP : {len(X)}")
print(f"\nAperçu des données :\n{df_acp.head()}")

# Standardisation (centrage-réduction) obligatoire pour l'ACP
# car les variables ont des unités différentes (%, kcal/cap/day, g/cap/day)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Transformation en DataFrame pour faciliter l'interprétation
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=variables_acp,
    index=df_acp.index
)

# Création du modèle ACP avec toutes les composantes
pca = PCA()
composantes_principales = pca.fit_transform(X_scaled)

# Création d'un DataFrame avec les composantes principales
cp_df = pd.DataFrame(
    composantes_principales,
    columns=[f'CP{i+1}' for i in range(len(variables_acp))],
    index=df_acp.index
)

# Ajout des identifiants (Pays, Année)
cp_df = pd.concat([identifiants.reset_index(drop=True), cp_df], axis=1)

print("\n" + "="*70)
print("RÉSULTATS DE L'ACP")
print("="*70)

# Variance expliquée par chaque composante
variance_expliquee = pca.explained_variance_ratio_
variance_cumulee = np.cumsum(variance_expliquee)

print("\n1. Variance expliquée par composante :")
for i, (var, cum_var) in enumerate(zip(variance_expliquee, variance_cumulee)):
    print(f"   CP{i+1} : {var*100:.2f}% (cumulé : {cum_var*100:.2f}%)")

# Contributions des variables à chaque composante (loadings)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'CP{i+1}' for i in range(len(variables_acp))],
    index=variables_acp
)

print("\n2. Contributions des variables (loadings) :")
print(loadings.round(3))

# On utilise la première composante principale comme score synthétique
# Plus la CP1 est élevée, meilleure est la sécurité alimentaire
cp_df['Score_Securite_Alimentaire'] = cp_df['CP1']

print("\n3. Score de sécurité alimentaire (CP1) par pays et année :")
print(cp_df[['Pays', 'Année', 'Score_Securite_Alimentaire']].sort_values(
    by=['Pays', 'Année']
))

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Éboulis des valeurs propres (scree plot)
axes[0, 0].bar(range(1, len(variance_expliquee) + 1), variance_expliquee * 100)
axes[0, 0].plot(range(1, len(variance_expliquee) + 1), variance_cumulee * 100, 
                'ro-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Composante principale')
axes[0, 0].set_ylabel('Variance expliquée (%)')
axes[0, 0].set_title('Éboulis des valeurs propres')
axes[0, 0].axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='80%')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Cercle des corrélations (CP1 et CP2)
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
axes[0, 1].add_artist(circle)
for i, var in enumerate(variables_acp):
    axes[0, 1].arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1],
                     head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    axes[0, 1].text(loadings.iloc[i, 0] * 1.15, loadings.iloc[i, 1] * 1.15,
                    var, fontsize=9, ha='center', va='center')
axes[0, 1].set_xlim(-1.2, 1.2)
axes[0, 1].set_ylim(-1.2, 1.2)
axes[0, 1].set_xlabel(f'CP1 ({variance_expliquee[0]*100:.1f}%)')
axes[0, 1].set_ylabel(f'CP2 ({variance_expliquee[1]*100:.1f}%)')
axes[0, 1].set_title('Cercle des corrélations (CP1-CP2)')
axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_aspect('equal')

# 3. Projection des individus (CP1 et CP2)
for pays in cp_df['Pays'].unique():
    data_pays = cp_df[cp_df['Pays'] == pays]
    axes[1, 0].scatter(data_pays['CP1'], data_pays['CP2'], 
                       label=pays, alpha=0.7, s=100)
axes[1, 0].set_xlabel(f'CP1 ({variance_expliquee[0]*100:.1f}%)')
axes[1, 0].set_ylabel(f'CP2 ({variance_expliquee[1]*100:.1f}%)')
axes[1, 0].set_title('Projection des observations (pays-années)')
axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 4. Évolution du score de sécurité alimentaire dans le temps
for pays in cp_df['Pays'].unique():
    data_pays = cp_df[cp_df['Pays'] == pays].sort_values('Année')
    axes[1, 1].plot(data_pays['Année'], data_pays['Score_Securite_Alimentaire'],
                    marker='o', label=pays, linewidth=2)
axes[1, 1].set_xlabel('Année')
axes[1, 1].set_ylabel('Score de sécurité alimentaire (CP1)')
axes[1, 1].set_title('Évolution temporelle du score de sécurité alimentaire')
axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Sauvegarde du DataFrame avec les scores
print("\n" + "="*70)
print("DataFrame avec les scores de sécurité alimentaire créé : 'cp_df'")
print("Variables disponibles : Pays, Année, CP1, CP2, CP3, CP4, Score_Securite_Alimentaire")
print("="*70)
