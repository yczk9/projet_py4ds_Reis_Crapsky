# On importe la bibliothèque permettant de récupérer les données disponible sur FAOSTAT
import faostat

def get_datas_fao (area,item,year) : 
    """
    Fonction que l'on appelera dans notre notebook pour récupérer les données pour les BRICSAM
    """
    # Creation du dataframe
    mypars = {'area': area, # on sélectionne uniquement les pays souahités
              'element': [6120], # 2 variables sont disponibles par indicateurs : valeur et intervalle de confiance. On ne garde que la première 
              'item': item, # code des indicateurs que l'on souhaite garder
              'year': year} # années séléctionnées

    df = faostat.get_data_df('FS', pars=mypars, strval=False) # FS correspond au jeu d'indicateurs sur la sécurité alimentaire
    
    # Nettoyage du dataframe
    ## On supprime les colonnes inutiles et on renomme celles restantes
    df = df.drop(['Domain Code', 'Domain', 'Area Code', 'Element Code', 'Element', 'Year Code','Item Code'], axis=1)
    df = df.rename(columns={
        'Item': 'Indicateur',
        'Area': 'Pays',
        'Year': 'Année',
        'Unit' : 'Unité',
        'Value' : 'Valeur'
        })

    
    # Modification du nom des indicateurs de sécurité alimentaire
    indicateurs_map = {
        'Average dietary energy supply adequacy (percent) (3-year average)': 
            "suffisance des apports énergétiques alimentaires moyens",
        'Dietary energy supply used in the estimation of the prevalence of undernourishment (kcal/cap/day) (3-year average)': 
            "disponibilité alimentaire par habitant",
        'Average protein supply (g/cap/day) (3-year average)': 
            "disponibilité protéiques moyenne"
    }
    df['Indicateur'] = df['Indicateur'].replace(indicateurs_map)
    
    return df

def format_annees (df) :
    """
    Permet de reformater la variable année.
    Pour l'instant la valeur de year s'écrit 'xxxx-yyyy'. Comme on ne veut garder que la première année ('xxxx'). 
    On ne garde que les 4 premiers caractères. On les passe ensuite en entier.
    """
    for year in df['Année'].unique():
        df.loc[df['Année'] == year, 'Année'] = int(year[:4])

    return df