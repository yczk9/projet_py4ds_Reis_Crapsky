import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def run_annual_regressions(df, col_pib='PIB par habitant ($)', col_score='Score_Securite_Alim_F1'):
    """
    Étape 1 : Calcul des régressions annuelles.
    Cette étape consiste à modéliser mathématiquement la relation entre la richesse économique (PIB) 
    et la sécurité alimentaire pour chaque année de l'étude. L'extraction systématique du 
    coefficient de détermination (R2) permet de quantifier la force de cette corrélation.
    """
    results_list = []
    years = sorted(df['Année'].unique())
    
    for year in years:
        df_year = df[df['Année'] == year].dropna(subset=[col_pib, col_score])
        
        if len(df_year) < 5: 
            continue
            
        X = df_year[[col_pib]].values
        y = df_year[col_score].values
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        results_list.append({
            'Année': year,
            'R2': r2_score(y, reg.predict(X)),
            'Pente': reg.coef_[0],
            'Intercept': reg.intercept_,
            'N_Pays': len(df_year)
        })
    
    return pd.DataFrame(results_list)

def plot_r2_evolution(df_results):
    """
    Étape 2 : Analyse de l'évolution de la corrélation (R2).
    Cette visualisation suit l'évolution du pouvoir explicatif du PIB sur la sécurité alimentaire. 
    Une baisse du R2 suggère que la richesse nationale devient un prédicteur moins déterminant 
    au profit d'autres facteurs structurels ou politiques.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_results['Année'], df_results['R2'], marker='o', color='teal', linewidth=2)
    ax.set_title("Évolution du pouvoir explicatif du PIB ($R^2$)")
    ax.set_ylabel("Coefficient de détermination ($R^2$)")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_threshold_effect(df_final, df_results, col_pib='PIB par habitant ($)', col_score='Score_Securite_Alim_F1'):
    """
    Étape 3 : Visualisation de l'effet de seuil.
    En projetant l'ensemble des données, cette étape cherche à identifier une possible 
    relation non-linéaire. On observe si, au-delà d'un certain niveau de richesse, 
    l'amélioration de la sécurité alimentaire stagne (effet de saturation).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df_final[col_pib], df_final[col_score], c=df_final['Année'], cmap='viridis', alpha=0.6)
    
    # Trace la droite de tendance pour la dernière année
    last_res = df_results.iloc[-1]
    x_range = np.array([df_final[col_pib].min(), df_final[col_pib].max()])
    y_range = last_res['Pente'] * x_range + last_res['Intercept']
    
    ax.plot(x_range, y_range, color='red', linestyle='--', label=f"Tendance {int(last_res['Année'])}")
    ax.set_xlabel("PIB par habitant ($)")
    ax.set_ylabel("Score de Sécurité Alimentaire (F1)")
    plt.colorbar(sc, label='Année')
    ax.legend()
    plt.show()

def print_regression_diagnostic(df_results):
    """
    Étape 4 : Diagnostic automatique des résultats.
    Cette dernière étape automatise l'interprétation en comparant les points de départ 
    et d'arrivée de l'étude. Elle permet de conclure sur la stabilité ou l'essoufflement 
    du lien entre développement économique et bien-être alimentaire.
    """
    r2_start, r2_end = df_results.iloc[0]['R2'], df_results.iloc[-1]['R2']
    delta = r2_end - r2_start

    print("-" * 50)
    print(f"DIAGNOSTIC : R2 initial = {r2_start:.2f} | R2 final = {r2_end:.2f}")
    if delta < -0.1:
        print("Résultat : Affaiblissement du lien (Hypothèse d'effet de seuil confirmée).")
    elif delta > 0.1:
        print("Résultat : Renforcement de la dépendance au PIB.")
    else:
        print("Résultat : Relation stable sur la période étudiée.")
    print("-" * 50)