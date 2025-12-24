import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output

import pandas as pd

def graph_evol_growth_rate (df, indicator) :
    """
    Cette fonction créée un graphique représentant l'évolution par pays d'un indicateur au cours du temps

    """
    for pays in df['Pays'].unique() :
        data =df.loc[df['Pays'] == pays]
        plt.plot(data['Année'],data[indicator],label=pays)

    plt.xlabel('Année')    
    plt.ylabel(indicator)
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(data['Année'].min(), data['Année'].max() + 1, 2))
    plt.tight_layout()
    plt.show()

def graph_evol_growth_rate (df, indicator) :
    """
    Cette fonction créée un graphique représentant l'évolution du taux de croissance annuel par pays d'un indicateur au cours du temps

    """
    indicator_growth_rate_pays = []

    for pays in df['Pays'].unique() :
        data =df.loc[df['Pays'] == pays]
        for year in data['Année'].unique() :
            indicator_growth_rate_pays.append((data[year+1,indicator]-data[year,indicator])/data[year,indicator])

        plt.plot(data['Année'],data[indicator],label=pays)

    plt.xlabel('Année')    
    plt.ylabel(indicator)
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(data['Année'].min(), data['Année'].max() + 1, 2))
    plt.tight_layout()
    plt.show()

def draw_tcam_table(df, indicator):
    """
    Affiche un tableau du TCAM par périodes avec un quadrillage et sans couleur.
    Périodes en lignes, Pays en colonnes.
    """
    periods = [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]
    countries = sorted(df['Pays'].unique())
    
    data_tcam = {}
    for start, end in periods:
        period_label = f"{start}-{end}"
        data_tcam[period_label] = {}
        
        for country in countries:
            val_start = df[(df['Pays'] == country) & (df['Année'] == start)][indicator].iloc[0]
            val_end = df[(df['Pays'] == country) & (df['Année'] == end)][indicator].iloc[0]
            n = end - start
            tcam = ((val_end / val_start)**(1/n) - 1) * 100
            data_tcam[period_label][country] = tcam
            
        avg_start = df[df['Année'] == start][indicator].mean()
        avg_end = df[df['Année'] == end][indicator].mean()
        tcam_avg = ((avg_end / avg_start)**(1/n) - 1) * 100
        data_tcam[period_label]['Moyenne BRICSAM'] = tcam_avg

    df_final = pd.DataFrame.from_dict(data_tcam, orient='index')
    
    # --- MISE EN FORME AVEC QUADRILLAGE ---
    # On définit les styles CSS pour les bordures
    table_styles = [
        # Bordures pour toutes les cellules et en-têtes
        {'selector': 'td, th', 'props': [('border', '1px solid black'), 
                                        ('padding', '10px'), 
                                        ('text-align', 'center')]},
        # Style spécifique pour les en-têtes (gris clair)
        {'selector': 'th', 'props': [('background-color', '1px solid black'), 
                                    ('font-weight', 'bold')]}
    ]
    
    styled_table = (df_final.style
                    .format("{:.2f}%")
                    .set_caption(f"Taux de Croissance Annuel Moyen : {indicator}")
                    .set_table_styles(table_styles)
                    # Force la fusion des bordures pour un rendu propre
                    .set_table_attributes('style="border-collapse: collapse; border: 1px solid black;"'))
    
    return styled_table


def play_bricsam_evolution(df, indicator_name, speed=0.3):
    """
    Anime un diagramme en bâtons horizontal directement dans le notebook.
    """
    # 1. Préparation des paramètres fixes
    years = sorted(df['Année'].unique())
    countries = df['Pays'].unique()
    # On fixe les couleurs pour que chaque pays garde la sienne
    colors = plt.cm.get_cmap('tab10', len(countries))
    color_map = dict(zip(countries, [colors(i) for i in range(len(countries))]))
    
    # On fixe l'échelle max de l'axe X pour éviter que le graphique ne "saute"
    max_val = df[indicator_name].max() * 1.1

    for year in years:
        # Filtrage et tri par valeur pour l'effet de "course"
        data_year = df[df['Année'] == year].sort_values(by=indicator_name)
        
        # Nettoyage de la sortie précédente
        clear_output(wait=True)
        
        # Création du graphique
        plt.figure(figsize=(10, 6))
        bars = plt.barh(data_year['Pays'], data_year[indicator_name], 
                        color=[color_map[c] for c in data_year['Pays']])
        
        # Esthétique
        plt.xlim(0, max_val)
        plt.title(f"Évolution : {indicator_name}\nAnnée : {year}", fontsize=15, fontweight='bold')
        plt.xlabel("Valeur")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Ajout des étiquettes de valeurs au bout des barres
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f' {width:,.0f}', 
                     va='center', ha='left', fontweight='bold')
        
        plt.show()
        
        # Pause pour laisser le temps de voir l'évolution
        time.sleep(speed)