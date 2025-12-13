import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats



# ============================================================================
# RÉCUPÉRATION DU PIB/HABITANT (CODE 22013)
# ============================================================================

print("\n" + "="*80)
print("RÉCUPÉRATION DES DONNÉES PIB/HABITANT")
print("="*80)

# Paramètres pour récupérer le PIB/habitant
mypars_pib = {
    'area': liste_pays,  # On utilise la même liste de pays
    'element': [6120],
    'item': ['22013'],  # Code pour le PIB/habitant
    'year': [2000, 2022]
}

# Récupération des données
df_pib = faostat.get_data_df('FS', pars=mypars_pib, strval=False)

# Nettoyage du dataframe PIB
df_pib = df_pib.drop(['Domain Code', 'Domain', 'Area Code', 'Element Code', 
                      'Element', 'Year Code', 'Item Code'], axis=1)
df_pib = df_pib.rename(columns={
    'Item': 'Indicateur',
    'Area': 'Pays',
    'Year': 'Année',
    'Unit': 'Unité',
    'Value': 'PIB_hab'
})

# Nettoyage des années
for year in df_pib['Année'].unique():
    df_pib.loc[df_pib['Année'] == year, 'Année'] = int(year[:4])

# Conversion des valeurs en float (au cas où il y aurait des '<' ou '>')
def convertir_valeur(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if val_str.startswith('<') or val_str.startswith('>'):
        try:
            return float(val_str[1:])
        except ValueError:
            return np.nan
    try:
        return float(val_str)
    except ValueError:
        return np.nan

df_pib['PIB_hab'] = df_pib['PIB_hab'].apply(convertir_valeur)

# Garder uniquement les colonnes nécessaires
df_pib = df_pib[['Pays', 'Année', 'PIB_hab']]

print(f"\nDonnées PIB/habitant récupérées : {len(df_pib)} observations")
print(f"Nombre de valeurs manquantes : {df_pib['PIB_hab'].isna().sum()}")
print(f"\nAperçu des données PIB :")
print(df_pib.head(10))

# ============================================================================
# FUSION DES DONNÉES : SCORE ACP + PIB/HABITANT
# ============================================================================

print("\n" + "="*80)
print("FUSION DES DONNÉES")
print("="*80)

# Fusion des deux dataframes sur Pays et Année
df_regression = cp_df.merge(df_pib, on=['Pays', 'Année'], how='inner')

# Suppression des lignes avec valeurs manquantes
df_regression = df_regression.dropna(subset=['Score_Securite_Alimentaire', 'PIB_hab'])

print(f"\nNombre d'observations pour la régression : {len(df_regression)}")
print(f"\nAperçu des données fusionnées :")
print(df_regression[['Pays', 'Année', 'Score_Securite_Alimentaire', 'PIB_hab']].head(10))

# Statistiques descriptives
print(f"\n--- Statistiques descriptives ---")
print(f"Score Sécurité Alimentaire : moyenne = {df_regression['Score_Securite_Alimentaire'].mean():.3f}, "
      f"std = {df_regression['Score_Securite_Alimentaire'].std():.3f}")
print(f"PIB/habitant : moyenne = {df_regression['PIB_hab'].mean():.0f}, "
      f"std = {df_regression['PIB_hab'].std():.0f}")

# ============================================================================
# RÉGRESSION LINÉAIRE : SCORE ~ PIB/HABITANT
# ============================================================================

print("\n" + "="*80)
print("RÉGRESSION LINÉAIRE : Score Sécurité Alimentaire = f(PIB/habitant)")
print("="*80)

# Préparation des données
X = df_regression[['PIB_hab']].values  # Variable explicative
y = df_regression['Score_Securite_Alimentaire'].values  # Variable à expliquer

# Ajustement du modèle
model = LinearRegression()
model.fit(X, y)

# Prédictions
y_pred = model.predict(X)

# Calcul des métriques
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
n = len(y)
p = 1  # Nombre de prédicteurs
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Test de significativité du coefficient
# Calcul de l'erreur standard du coefficient
residuals = y - y_pred
mse = np.sum(residuals**2) / (n - 2)
X_centered = X - X.mean()
var_beta = mse / np.sum(X_centered**2)
se_beta = np.sqrt(var_beta)
t_stat = model.coef_[0] / se_beta
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))

# Affichage des résultats
print(f"\n--- RÉSULTATS DE LA RÉGRESSION ---")
print(f"\nÉquation : Score = {model.intercept_:.4f} + {model.coef_[0]:.6f} × PIB/hab")
print(f"\nCoefficient de régression :")
print(f"  - Pente (β₁) : {model.coef_[0]:.6f}")
print(f"  - Erreur standard : {se_beta:.6f}")
print(f"  - Statistique t : {t_stat:.3f}")
print(f"  - p-value : {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'}")
print(f"\nOrdonnée à l'origine (β₀) : {model.intercept_:.4f}")
print(f"\nQualité de l'ajustement :")
print(f"  - R² : {r2:.4f} ({r2*100:.2f}% de variance expliquée)")
print(f"  - R² ajusté : {r2_adj:.4f}")
print(f"  - RMSE : {rmse:.4f}")
print(f"  - N observations : {n}")

print(f"\n--- INTERPRÉTATION ---")
print(f"Une augmentation de 1000$ du PIB/habitant est associée à une augmentation")
print(f"de {model.coef_[0]*1000:.4f} du score de sécurité alimentaire.")

# ============================================================================
# VISUALISATIONS DE LA RÉGRESSION
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('RÉGRESSION LINÉAIRE : SCORE SÉCURITÉ ALIMENTAIRE ~ PIB/HABITANT', 
             fontsize=16, fontweight='bold')

# 1. Nuage de points avec droite de régression
ax = axes[0, 0]
colors_pays = plt.cm.tab10(np.linspace(0, 1, len(df_regression['Pays'].unique())))
pays_colors = {pays: color for pays, color in zip(df_regression['Pays'].unique(), colors_pays)}

for pays in df_regression['Pays'].unique():
    data_pays = df_regression[df_regression['Pays'] == pays]
    ax.scatter(data_pays['PIB_hab'], data_pays['Score_Securite_Alimentaire'],
               label=pays, alpha=0.7, s=80, color=pays_colors[pays], edgecolors='black')

# Droite de régression
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
ax.plot(X_line, y_line, 'r--', linewidth=3, label='Régression linéaire', alpha=0.8)

ax.set_xlabel('PIB/habitant (USD constant 2017 PPP)', fontweight='bold', fontsize=11)
ax.set_ylabel('Score Sécurité Alimentaire (CP1)', fontweight='bold', fontsize=11)
ax.set_title(f'Régression Linéaire\nR² = {r2:.4f}', fontweight='bold', fontsize=12)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3, linestyle=':')

# Ajout de l'équation sur le graphique
equation_text = f'Score = {model.intercept_:.2f} + {model.coef_[0]:.5f} × PIB/hab\nR² = {r2:.4f}, p < {p_value:.4f}'
ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 2. Résidus vs Valeurs prédites
ax = axes[0, 1]
residuals = y - y_pred
ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Valeurs prédites', fontweight='bold')
ax.set_ylabel('Résidus', fontweight='bold')
ax.set_title('Résidus vs Valeurs Prédites\n(Homoscédasticité)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle=':')

# 3. Q-Q plot (normalité des résidus)
ax = axes[0, 2]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot\n(Normalité des résidus)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle=':')

# 4. Distribution des résidus
ax = axes[1, 0]
ax.hist(residuals, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Moyenne')
ax.set_xlabel('Résidus', fontweight='bold')
ax.set_ylabel('Fréquence', fontweight='bold')
ax.set_title('Distribution des Résidus', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y', linestyle=':')

# 5. Valeurs observées vs prédites
ax = axes[1, 1]
ax.scatter(y, y_pred, alpha=0.6, edgecolors='black', s=60)
# Ligne y=x (prédiction parfaite)
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x (parfait)')
ax.set_xlabel('Scores Observés', fontweight='bold')
ax.set_ylabel('Scores Prédits', fontweight='bold')
ax.set_title(f'Observé vs Prédit\nR² = {r2:.4f}', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, linestyle=':')

# 6. Régression par pays (trajectoires)
ax = axes[1, 2]
for pays in df_regression['Pays'].unique():
    data_pays = df_regression[df_regression['Pays'] == pays].sort_values('Année')
    if len(data_pays) > 1:
        ax.plot(data_pays['PIB_hab'], data_pays['Score_Securite_Alimentaire'],
                marker='o', label=pays, linewidth=2, markersize=6, 
                color=pays_colors[pays], alpha=0.7)

# Droite de régression globale
ax.plot(X_line, y_line, 'k--', linewidth=3, label='Régression globale', alpha=0.5)
ax.set_xlabel('PIB/habitant', fontweight='bold')
ax.set_ylabel('Score Sécurité Alimentaire', fontweight='bold')
ax.set_title('Trajectoires par Pays', fontweight='bold', fontsize=12)
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
fig.savefig('figure4_regression_pib_securite.png', dpi=300, bbox_inches='tight')
print("\n Figure 4 sauvegardée : figure4_regression_pib_securite.png")
plt.close(fig)

# ============================================================================
# ANALYSE PAR PAYS
# ============================================================================

print("\n" + "="*80)
print("RÉGRESSION PAR PAYS (ANALYSES INDIVIDUELLES)")
print("="*80)

resultats_pays = []

for pays in df_regression['Pays'].unique():
    data_pays = df_regression[df_regression['Pays'] == pays]
    
    if len(data_pays) >= 3:  # On a besoin d'au moins 3 observations
        X_pays = data_pays[['PIB_hab']].values
        y_pays = data_pays['Score_Securite_Alimentaire'].values
        
        # Régression pour ce pays
        model_pays = LinearRegression()
        model_pays.fit(X_pays, y_pays)
        y_pred_pays = model_pays.predict(X_pays)
        r2_pays = r2_score(y_pays, y_pred_pays)
        
        resultats_pays.append({
            'Pays': pays,
            'Pente': model_pays.coef_[0],
            'Intercept': model_pays.intercept_,
            'R²': r2_pays,
            'N_obs': len(data_pays)
        })

# Création d'un DataFrame avec les résultats
df_resultats_pays = pd.DataFrame(resultats_pays).sort_values('R²', ascending=False)

print("\n--- Résultats des régressions par pays ---")
print(df_resultats_pays.to_string(index=False))

# ============================================================================
# TABLEAU RÉCAPITULATIF FINAL
# ============================================================================

print("\n" + "="*80)
print("TABLEAU RÉCAPITULATIF : SÉCURITÉ ALIMENTAIRE ET PIB/HABITANT")
print("="*80)

# Calcul des moyennes par pays
recap = df_regression.groupby('Pays').agg({
    'Score_Securite_Alimentaire': ['mean', 'std'],
    'PIB_hab': ['mean', 'std']
}).round(3)

recap.columns = ['Score_Moyen', 'Score_Std', 'PIB_Moyen', 'PIB_Std']
recap = recap.sort_values('Score_Moyen', ascending=False)

print(recap.to_string())
print("="*80)

# ============================================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================================

# Sauvegarde du dataframe avec PIB
df_regression.to_csv('donnees_regression_securite_pib.csv', index=False)
print("\n Données sauvegardées : donnees_regression_securite_pib.csv")

print("\n" + "="*80)
print("ANALYSE DE RÉGRESSION TERMINÉE")
print("="*80)
print("Fichiers créés :")
print("  1. figure4_regression_pib_securite.png")
print("  2. donnees_regression_securite_pib.csv")
print("="*80)
