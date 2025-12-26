import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import clear_output

import pandas as pd

def graph_evol(df, indicator) :
    """
    Cette fonction créée un graphique représentant l'évolution par pays 
    ainsi que de la moyenne des pays d'un indicateur au cours du temps

    """
    # Graphe pour chaque pays
    for pays in df['Pays'].unique() :
        data =df.loc[df['Pays'] == pays]
        plt.plot(data['Année'],data[indicator],label=pays)

    # Graphe pour la moyenne des pays
    plt.plot(df['Année'].unique(),df.groupby('Année')[indicator].mean(), label='Moyenne BRICSAM')

    plt.xlabel('Année')    
    plt.ylabel(indicator)
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(data['Année'].min(), data['Année'].max() + 1, 2))
    plt.xlim(2000,2020)
    plt.tight_layout()
    plt.show()

def graph_evol_TCA (df, indicator) :
    """
    Cette fonction créée un graphique représentant l'évolution du taux de croissance annuel par pays 
    ainsi que pour la moyenne des pays d'un indicateur au cours du temps

    """
    years = sorted(df['Année'].unique())
    years.pop() # On 'exclue' 2020 mais en réalité on exclue 2000 car on calcule entre l'année n et l'année n+1

    # Graphe pour chaque pays
    for pays in df['Pays'].unique() :
        data =df.loc[df['Pays'] == pays]
        data_TCA =[0] # on va considérer que le taux de croissance en 2000 est nul
        for year in years : 
            TCA = (df[(df['Pays'] == pays) & (df['Année'] == year+1)][indicator].iloc[0]-df[(df['Pays'] == pays) & (df['Année'] == year)][indicator].iloc[0])/df[(df['Pays'] == pays) & (df['Année'] == year)][indicator].iloc[0]
            data_TCA.append(TCA*100)
        plt.plot(data['Année'],data_TCA,label=pays)

    # Graphe pour la moyenne des pays
    data_TCA =[0] 
    for year in years : 
        TCA = (df[(df['Année'] == year+1)][indicator].iloc[0].mean()-df[(df['Année'] == year)][indicator].iloc[0].mean())/df[(df['Année'] == year)][indicator].iloc[0].mean()
        data_TCA.append(TCA*100)  
    plt.plot(df['Année'].unique(),data_TCA, label='Moyenne BRICSAM')

    plt.xlabel('Année')    
    plt.ylabel(indicator)
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(data['Année'].min(), data['Année'].max() + 1, 2))
    plt.xlim(2000,2020)
    plt.tight_layout()
    plt.show()

def draw_tcam_table(df, indicator):
    """
    Affiche un tableau du taux de croissance annuelle moyen (TCAM) par périodes de 5 ans pour chaque pays et pour la moyenne des pays.
   
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
            tcam = ((val_end - val_start)/(val_start)) * 100 # ((PIBn- PIBn-1) / PIBn-1) * 100
            data_tcam[period_label][country] = tcam
            
        avg_start = df[df['Année'] == start][indicator].mean()
        avg_end = df[df['Année'] == end][indicator].mean()
        tcam_avg = ((avg_end - avg_start)/(avg_start)) * 100
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
        #{'selector': 'th', 'props': [('background-color', '1px solid black'), 
                                   # ('font-weight', 'bold')]}
    ]
    
    styled_table = (df_final.style
                    .format("{:.2f}%")
                    .set_caption(f"Taux de Croissance Annuel Moyen : {indicator}")
                    .set_table_styles(table_styles)
                    # Force la fusion des bordures pour un rendu propre
                    .set_table_attributes('style="border-collapse: collapse; border: 1px solid black;"'))
    
    return styled_table


def play_bricsam_evolution(df, indicator, speed):
    """
    Diagramme en batons animé décrivant l'évolution par année d'un indicateur. 

    """
    # 1. Préparation des paramètres fixes
    years = sorted(df['Année'].unique())
    countries = df['Pays'].unique()
    # On fixe les couleurs pour que chaque pays garde la sienne
    colors = plt.cm.get_cmap('tab10', len(countries))
    color_map = dict(zip(countries, [colors(i) for i in range(len(countries))]))
    
    # On fixe l'échelle max de l'axe X pour éviter que le graphique ne "saute"
    max_val = df[indicator].max() * 1.1

    for year in years:
        # Filtrage et tri par valeur pour l'effet de "course" entre pays
        data_year = df[df['Année'] == year].sort_values(by=indicator)
        
        # Nettoyage de la sortie précédente (on enlève le baton de l'année précédente)
        clear_output(wait=True)
        
        # Création du graphique
        plt.figure(figsize=(10, 6))
        bars = plt.barh(data_year['Pays'], data_year[indicator], 
                        color=[color_map[c] for c in data_year['Pays']])
        
        # Esthétique
        plt.xlim(0, max_val)
        plt.title(f"Évolution : {indicator}\nAnnée : {year}", fontsize=15, fontweight='bold')
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