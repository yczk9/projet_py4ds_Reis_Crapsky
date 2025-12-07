import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# On importe la bibliothèque permettant de récupérer les données disponible sur FAOSTAT
import faostat

def get_datas_fao(liste_pays):
    """
    Fonction pour récupérer les données FAO sur la sécurité alimentaire
    """
    mypars = {
        'area': liste_pays,
        'element': [6120],
        'item': ['21010', '22000', '21013', '21004'],
        'year': [2000, 2022]
    }
    df = faostat.get_data_df('FS', pars=mypars, strval=False)
    
    # Nettoyage des colonnes
    df = df.drop(['Domain Code', 'Domain', 'Area Code', 'Element Code', 
                  'Element', 'Year Code', 'Item Code'], axis=1)
    df = df.rename(columns={
        'Item': 'Indicateur',
        'Area': 'Pays',
        'Year': 'Année',
        'Unit': 'Unité',
        'Value': 'Valeur'
    })

    # Nettoyage des années
    for year in df['Année'].unique():
        df.loc[df['Année'] == year, 'Année'] = int(year[:4])

    # Renommage des indicateurs
    indicateurs_map = {
        'Average dietary energy supply adequacy (percent) (3-year average)': 
            "suffisance des apports énergétiques alimentaires moyens",
        'Dietary energy supply used in the estimation of the prevalence of undernourishment (kcal/cap/day) (3-year average)': 
            "disponibilité alimentaire par habitant",
        'Average protein supply (g/cap/day) (3-year average)': 
            "disponibilité protéiques moyenne",
        'Prevalence of undernourishment (percent) (3-year average)': 
            "prévalence de la sous alimentation"
    }
    
    df['Indicateur'] = df['Indicateur'].replace(indicateurs_map)
    
    return df


def nettoyer_valeurs(df):
    """
    Nettoie les valeurs de la colonne 'Valeur' pour gérer les cas comme '<2.5'
    """
    def convertir_valeur(val):
        """Convertit une valeur en float, en gérant les cas spéciaux"""
        if pd.isna(val):
            return np.nan
        
        # Si c'est déjà un nombre
        if isinstance(val, (int, float)):
            return float(val)
        
        # Si c'est une chaîne
        val_str = str(val).strip()
        
        # Gestion des cas avec '<' (inférieur à)
        if val_str.startswith('<'):
            # On prend la valeur après '<' comme approximation
            return float(val_str[1:])
        
        # Gestion des cas avec '>' (supérieur à)
        if val_str.startswith('>'):
            return float(val_str[1:])
        
        # Tentative de conversion directe
        try:
            return float(val_str)
        except ValueError:
            return np.nan
    
    df['Valeur'] = df['Valeur'].apply(convertir_valeur)
    return df


# ============================================================================
# RÉCUPÉRATION ET NETTOYAGE DES DONNÉES
# ============================================================================

liste_pays = [202, 185, 138, 100, 351, 21]  # Codes des pays BRICSAM
df = get_datas_fao(liste_pays)

# ÉTAPE CRITIQUE : Nettoyage des valeurs
print("Avant nettoyage :")
print(df['Valeur'].dtype)
print(df['Valeur'].head(10))

df = nettoyer_valeurs(df)

print("\nAprès nettoyage :")
print(df['Valeur'].dtype)
print(df['Valeur'].head(10))
print(f"\nNombre de valeurs manquantes : {df['Valeur'].isna().sum()}")

# ============================================================================
# PRÉPARATION DES DONNÉES POUR L'ACP
# ============================================================================

# Création d'un pivot pour avoir les indicateurs en colonnes
df_pivot = df.pivot_table(
    index=['Pays', 'Année'],
    columns='Indicateur',
    values='Valeur',
    aggfunc='mean'  # On spécifie explicitement la fonction d'agrégation
).reset_index()

# Renommage des colonnes pour plus de lisibilité
df_pivot.columns.name = None

print("\nAperçu du pivot :")
print(df_pivot.head())
print(f"\nNombre de lignes : {len(df_pivot)}")

# Inversion de la prévalence de la sous-alimentation (21004)
# Car elle évolue dans le sens opposé aux autres indicateurs
df_pivot['prévalence de la sous alimentation'] = -df_pivot['prévalence de la sous alimentation']

# ============================================================================
# SÉLECTION DES VARIABLES POUR L'ACP
# ============================================================================

# Variables retenues pour l'ACP (nos 4 indicateurs)
variables_acp = [
    'suffisance des apports énergétiques alimentaires moyens',
    'disponibilité alimentaire par habitant',
    'disponibilité protéiques moyenne',
    'prévalence de la sous alimentation'  # Maintenant inversée
]

# Extraction des données pour l'ACP (suppression des valeurs manquantes)
df_acp = df_pivot[['Pays', 'Année'] + variables_acp].dropna()

print(f"\nNombre d'observations pour l'ACP : {len(df_acp)}")
print(f"\nAperçu des données :\n{df_acp.head()}")

# Vérification qu'on a bien des données
if len(df_acp) == 0:
    raise ValueError("Aucune donnée disponible après suppression des valeurs manquantes!")

# Séparation des identifiants et des variables quantitatives
X = df_acp[variables_acp]
identifiants = df_acp[['Pays', 'Année']]

# ============================================================================
# STANDARDISATION DES DONNÉES
# ============================================================================

# Standardisation (centrage-réduction) obligatoire pour l'ACP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Transformation en DataFrame pour faciliter l'interprétation
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=variables_acp,
    index=df_acp.index
)

# ============================================================================
# RÉALISATION DE L'ACP
# ============================================================================

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

# ============================================================================
# ANALYSE DES RÉSULTATS
# ============================================================================

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

# Score de sécurité alimentaire
cp_df['Score_Securite_Alimentaire'] = cp_df['CP1']

print("\n3. Score de sécurité alimentaire (CP1) par pays et année :")
print(cp_df[['Pays', 'Année', 'Score_Securite_Alimentaire']].sort_values(
    by=['Pays', 'Année']
))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# VISUALISATIONS AVANCÉES POUR L'ACP SÉCURITÉ ALIMENTAIRE
# ============================================================================

# Configuration générale
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

# Correction du problème de NaN dans cp_df
cp_df = pd.concat([identifiants.reset_index(drop=True), 
                   pd.DataFrame(composantes_principales, 
                               columns=[f'CP{i+1}' for i in range(len(variables_acp))])], 
                  axis=1)
cp_df['Score_Securite_Alimentaire'] = cp_df['CP1']

# Suppression des lignes avec NaN
cp_df = cp_df.dropna()

print(f"Données nettoyées : {len(cp_df)} observations")

# ============================================================================
# FIGURE 1 : VUE D'ENSEMBLE DE L'ACP
# ============================================================================

fig1 = plt.figure(figsize=(18, 10))
gs = fig1.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1.1 Éboulis des valeurs propres avec pourcentages
ax1 = fig1.add_subplot(gs[0, 0])
x_pos = range(1, len(variance_expliquee) + 1)
bars = ax1.bar(x_pos, variance_expliquee * 100, color='steelblue', alpha=0.7, edgecolor='black')
ax1.plot(x_pos, variance_cumulee * 100, 'ro-', linewidth=2.5, markersize=10, label='Cumulé')

# Ajout des valeurs sur les barres
for i, (bar, val) in enumerate(zip(bars, variance_expliquee * 100)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Seuil 90%')
ax1.set_xlabel('Composante Principale', fontweight='bold')
ax1.set_ylabel('Variance Expliquée (%)', fontweight='bold')
ax1.set_title('Éboulis des Valeurs Propres\n(CP1 explique 91.5% !)', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_xticks(x_pos)

# 1.2 Cercle des corrélations amélioré
ax2 = fig1.add_subplot(gs[0, 1])
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=2)
ax2.add_artist(circle)

# Couleurs par variable
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i, (var, color) in enumerate(zip(variables_acp, colors)):
    # Flèche
    ax2.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1],
              head_width=0.05, head_length=0.05, fc=color, ec=color, linewidth=2.5, alpha=0.8)
    # Label avec fond blanc
    ax2.text(loadings.iloc[i, 0] * 1.18, loadings.iloc[i, 1] * 1.18,
             var.split()[0] + '\n' + ' '.join(var.split()[1:3]), 
             fontsize=8, ha='center', va='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

ax2.set_xlim(-1.3, 1.3)
ax2.set_ylim(-1.3, 1.3)
ax2.set_xlabel(f'CP1 ({variance_expliquee[0]*100:.1f}%)', fontweight='bold', fontsize=11)
ax2.set_ylabel(f'CP2 ({variance_expliquee[1]*100:.1f}%)', fontweight='bold', fontsize=11)
ax2.set_title('Cercle des Corrélations\n(Toutes les variables contribuent positivement à CP1)', 
              fontweight='bold', fontsize=12)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.8)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_aspect('equal')

# 1.3 Contributions des variables (barplot horizontal)
ax3 = fig1.add_subplot(gs[0, 2])
contributions_cp1 = loadings['CP1'].sort_values(ascending=True)
bars = ax3.barh(range(len(contributions_cp1)), contributions_cp1.values, color=colors, edgecolor='black')
ax3.set_yticks(range(len(contributions_cp1)))
ax3.set_yticklabels([v.split()[0] + '\n' + ' '.join(v.split()[1:3]) for v in contributions_cp1.index], 
                     fontsize=9)
ax3.set_xlabel('Contribution à CP1', fontweight='bold')
ax3.set_title('Contributions des Variables à CP1\n(Toutes positives = cohérence)', 
              fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3, axis='x', linestyle=':')

# Ajout des valeurs
for i, (bar, val) in enumerate(zip(bars, contributions_cp1.values)):
    ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontweight='bold', fontsize=9)

# 1.4 Distribution des scores par pays
ax4 = fig1.add_subplot(gs[1, :])
pays_list = cp_df['Pays'].unique()
positions = []
data_violin = []

for i, pays in enumerate(pays_list):
    scores = cp_df[cp_df['Pays'] == pays]['Score_Securite_Alimentaire'].values
    data_violin.append(scores)
    positions.append(i)

# Violin plot
parts = ax4.violinplot(data_violin, positions=positions, widths=0.7, 
                       showmeans=True, showextrema=True)

# Coloration
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')

# Ajout de points individuels
for i, (pos, scores) in enumerate(zip(positions, data_violin)):
    y_jitter = scores + np.random.normal(0, 0.02, len(scores))
    ax4.scatter([pos] * len(scores), y_jitter, alpha=0.4, s=30, color='darkblue')

ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Moyenne globale', alpha=0.7)
ax4.set_xticks(positions)
ax4.set_xticklabels(pays_list, rotation=45, ha='right')
ax4.set_ylabel('Score de Sécurité Alimentaire (CP1)', fontweight='bold', fontsize=11)
ax4.set_title('Distribution des Scores par Pays (2000-2022)\nViolin Plot + Points Individuels', 
              fontweight='bold', fontsize=13)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y', linestyle=':')

plt.suptitle('ANALYSE EN COMPOSANTES PRINCIPALES - SÉCURITÉ ALIMENTAIRE BRICSAM', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# ============================================================================
# FIGURE 2 : ÉVOLUTIONS TEMPORELLES
# ============================================================================

fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('ÉVOLUTION TEMPORELLE DE LA SÉCURITÉ ALIMENTAIRE (2000-2022)', 
              fontsize=16, fontweight='bold')

# 2.1 Évolution des scores - Une ligne par pays
ax = axes[0, 0]
for pays in cp_df['Pays'].unique():
    data_pays = cp_df[cp_df['Pays'] == pays].sort_values('Année')
    ax.plot(data_pays['Année'], data_pays['Score_Securite_Alimentaire'],
            marker='o', label=pays, linewidth=2.5, markersize=6)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
ax.set_xlabel('Année', fontweight='bold')
ax.set_ylabel('Score CP1', fontweight='bold')
ax.set_title('Trajectoires Individuelles', fontweight='bold', fontsize=12)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3, linestyle=':')

# 2.2 Taux de croissance moyen du score
ax = axes[0, 1]
taux_croissance = {}
for pays in cp_df['Pays'].unique():
    data_pays = cp_df[cp_df['Pays'] == pays].sort_values('Année')
    if len(data_pays) > 1:
        score_debut = data_pays['Score_Securite_Alimentaire'].iloc[0]
        score_fin = data_pays['Score_Securite_Alimentaire'].iloc[-1]
        nb_annees = data_pays['Année'].iloc[-1] - data_pays['Année'].iloc[0]
        if nb_annees > 0:
            taux = ((score_fin - score_debut) / nb_annees) * 100
            taux_croissance[pays] = taux

taux_df = pd.Series(taux_croissance).sort_values()
colors_bar = ['green' if x > 0 else 'red' for x in taux_df.values]
bars = ax.barh(range(len(taux_df)), taux_df.values, color=colors_bar, edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(taux_df)))
ax.set_yticklabels(taux_df.index)
ax.set_xlabel('Taux de croissance annuel (%)', fontweight='bold')
ax.set_title('Progression Annuelle Moyenne\n(Vert=amélioration, Rouge=dégradation)', 
             fontweight='bold', fontsize=11)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(True, alpha=0.3, axis='x', linestyle=':')

# Ajout des valeurs
for bar, val in zip(bars, taux_df.values):
    ax.text(val + (0.5 if val > 0 else -0.5), bar.get_y() + bar.get_height()/2, 
            f'{val:.2f}%', va='center', ha='left' if val > 0 else 'right', 
            fontweight='bold', fontsize=9)

# 2.3 Heatmap des scores par année
ax = axes[0, 2]
pivot_scores = cp_df.pivot_table(index='Pays', columns='Année', 
                                  values='Score_Securite_Alimentaire')
sns.heatmap(pivot_scores, annot=False, cmap='RdYlGn', center=0, 
            cbar_kws={'label': 'Score CP1'}, ax=ax, linewidths=0.5)
ax.set_title('Heatmap des Scores\n(Vert=bon, Rouge=mauvais)', fontweight='bold', fontsize=12)
ax.set_xlabel('Année', fontweight='bold')
ax.set_ylabel('Pays', fontweight='bold')

# 2.4 Comparaison début vs fin de période
ax = axes[1, 0]
# Trouver les années disponibles
annees_disponibles = sorted(cp_df['Année'].unique())
annee_debut = annees_disponibles[0]
annee_fin = annees_disponibles[-1]

scores_debut = cp_df[cp_df['Année'] == annee_debut].set_index('Pays')['Score_Securite_Alimentaire']
scores_fin = cp_df[cp_df['Année'] == annee_fin].set_index('Pays')['Score_Securite_Alimentaire']

# Garder uniquement les pays présents dans les deux années
pays_communs = scores_debut.index.intersection(scores_fin.index)

if len(pays_communs) > 0:
    scores_debut = scores_debut.loc[pays_communs]
    scores_fin = scores_fin.loc[pays_communs]
    
    x = np.arange(len(pays_communs))
    width = 0.35
    bars1 = ax.bar(x - width/2, scores_debut.values, width, label=f'{int(annee_debut)}', 
                   color='lightcoral', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, scores_fin.values, width, label=f'{int(annee_fin)}', 
                   color='lightgreen', edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(pays_communs, rotation=45, ha='right')
    ax.set_ylabel('Score CP1', fontweight='bold')
    ax.set_title(f'Comparaison {int(annee_debut)} vs {int(annee_fin)}', fontweight='bold', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
else:
    ax.text(0.5, 0.5, 'Données insuffisantes', ha='center', va='center', 
            transform=ax.transAxes, fontsize=12)
    ax.set_title(f'Comparaison {int(annee_debut)} vs {int(annee_fin)}', fontweight='bold', fontsize=12)

# 2.5 Écart-type par pays (volatilité)
ax = axes[1, 1]
volatilite = cp_df.groupby('Pays')['Score_Securite_Alimentaire'].std().sort_values()
bars = ax.barh(range(len(volatilite)), volatilite.values, 
               color='orange', edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(volatilite)))
ax.set_yticklabels(volatilite.index)
ax.set_xlabel('Écart-type du Score', fontweight='bold')
ax.set_title('Volatilité de la Sécurité Alimentaire\n(Plus élevé = plus instable)', 
             fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='x', linestyle=':')

# 2.6 Classement final (dernière année disponible)
ax = axes[1, 2]
annee_finale = cp_df['Année'].max()
scores_finaux = cp_df[cp_df['Année'] == annee_finale].sort_values('Score_Securite_Alimentaire')

if len(scores_finaux) > 0:
    colors_rank = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(scores_finaux)))
    bars = ax.barh(range(len(scores_finaux)), scores_finaux['Score_Securite_Alimentaire'].values,
                   color=colors_rank, edgecolor='black')
    ax.set_yticks(range(len(scores_finaux)))
    ax.set_yticklabels(scores_finaux['Pays'].values)
    ax.set_xlabel(f'Score CP1 en {int(annee_finale)}', fontweight='bold')
    ax.set_title(f'Classement Final {int(annee_finale)}\n(Du moins bon au meilleur)', 
                 fontweight='bold', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='x', linestyle=':')
    
    # Ajout des valeurs
    for bar, val in zip(bars, scores_finaux['Score_Securite_Alimentaire'].values):
        ax.text(val + (0.1 if val > 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', ha='left' if val > 0 else 'right', 
                fontweight='bold', fontsize=9)
else:
    ax.text(0.5, 0.5, 'Données insuffisantes', ha='center', va='center', 
            transform=ax.transAxes, fontsize=12)
    ax.set_title(f'Classement Final {int(annee_finale)}', fontweight='bold', fontsize=12)

plt.tight_layout()
# Sauvegarde de la figure au lieu de l'afficher
fig1.savefig('figure1_vue_ensemble_acp.png', dpi=300, bbox_inches='tight')
print("\n✅ Figure 1 sauvegardée : figure1_vue_ensemble_acp.png")
plt.close(fig1)

# ============================================================================
# FIGURE 3 : ANALYSE MULTIVARIÉE
# ============================================================================

fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle('ANALYSE MULTIVARIÉE - PROJECTION ET CORRÉLATIONS', 
              fontsize=16, fontweight='bold')

# 3.1 Projection des individus colorée par pays
ax = axes[0, 0]
for pays in cp_df['Pays'].unique():
    data_pays = cp_df[cp_df['Pays'] == pays]
    ax.scatter(data_pays['CP1'], data_pays['CP2'], 
               label=pays, alpha=0.7, s=100, edgecolors='black', linewidth=1)
ax.set_xlabel(f'CP1 ({variance_expliquee[0]*100:.1f}%)', fontweight='bold')
ax.set_ylabel(f'CP2 ({variance_expliquee[1]*100:.1f}%)', fontweight='bold')
ax.set_title('Projection sur CP1-CP2\npar Pays', fontweight='bold', fontsize=12)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3, linestyle=':')

# 3.2 Projection colorée par année
ax = axes[0, 1]
years = cp_df['Année'].values
scatter = ax.scatter(cp_df['CP1'], cp_df['CP2'], 
                     c=years, cmap='viridis', s=100, alpha=0.7, 
                     edgecolors='black', linewidth=1)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Année', fontweight='bold')
ax.set_xlabel(f'CP1 ({variance_expliquee[0]*100:.1f}%)', fontweight='bold')
ax.set_ylabel(f'CP2 ({variance_expliquee[1]*100:.1f}%)', fontweight='bold')
ax.set_title('Projection sur CP1-CP2\npar Année', fontweight='bold', fontsize=12)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
ax.grid(True, alpha=0.3, linestyle=':')

# 3.3 Matrice de corrélation des variables originales
ax = axes[1, 0]
corr_matrix = X.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={'label': 'Corrélation'},
            mask=mask, ax=ax)
ax.set_title('Matrice de Corrélation\ndes Variables Originales', 
             fontweight='bold', fontsize=12)

# 3.4 Biplot (individus + variables)
ax = axes[1, 1]
# Points (individus) - échelle réduite
scale_factor = 3
for pays in cp_df['Pays'].unique():
    data_pays = cp_df[cp_df['Pays'] == pays]
    ax.scatter(data_pays['CP1'] / scale_factor, data_pays['CP2'] / scale_factor, 
               alpha=0.3, s=50, label=pays)

# Flèches (variables)
for i, var in enumerate(variables_acp):
    ax.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1],
             head_width=0.05, head_length=0.05, fc='red', ec='red', 
             linewidth=2.5, alpha=0.8)
    ax.text(loadings.iloc[i, 0] * 1.15, loadings.iloc[i, 1] * 1.15,
            var.split()[0], fontsize=9, ha='center', va='center', 
            fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_xlabel(f'CP1 ({variance_expliquee[0]*100:.1f}%)', fontweight='bold')
ax.set_ylabel(f'CP2 ({variance_expliquee[1]*100:.1f}%)', fontweight='bold')
ax.set_title('Biplot : Individus + Variables\n(Points=observations, Flèches=variables)', 
             fontweight='bold', fontsize=12)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, linestyle=':')
ax.legend(loc='upper right', fontsize=7)

plt.tight_layout()
# Sauvegarde de la figure
fig2.savefig('figure2_evolutions_temporelles.png', dpi=300, bbox_inches='tight')
print("✅ Figure 2 sauvegardée : figure2_evolutions_temporelles.png")
plt.close(fig2)

# ============================================================================
# TABLEAU RÉCAPITULATIF
# ============================================================================

print("\n" + "="*80)
print("TABLEAU RÉCAPITULATIF DES STATISTIQUES PAR PAYS")
print("="*80)

stats_pays = cp_df.groupby('Pays')['Score_Securite_Alimentaire'].agg([
    ('Score_Moyen', 'mean'),
    ('Score_Min', 'min'),
    ('Score_Max', 'max'),
    ('Écart-type', 'std'),
    ('Nb_Obs', 'count')
]).round(3)

# Ajout du score de la dernière année disponible
annee_finale = cp_df['Année'].max()
score_final = cp_df[cp_df['Année'] == annee_finale].set_index('Pays')['Score_Securite_Alimentaire']
stats_pays[f'Score_{int(annee_finale)}'] = score_final
stats_pays = stats_pays.sort_values(f'Score_{int(annee_finale)}', ascending=False)

print(stats_pays.to_string())
print("="*80)
