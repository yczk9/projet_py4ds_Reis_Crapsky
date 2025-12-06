# On importe la bibliothèque permettant de récupérer les données disponible sur FAOSTAT
import faostat

def get_datas_fao (liste_pays) : # Fonction que l'on appelera dans notre notebook pour récupérer les données pour les BRICSAM
    mypars = {'area': liste_pays, # on sélectionne uniquement les pays souahités
              'element': [6120], # 2 variables sont disponibles par indicateurs : valeur et intervalle de confiance. On ne garde que la première 
              'item': ['21010','22000','21013','21004'], # code des indicateurs que l'on souhaite garder
              'year': [2000, 2022]} # années séléctionnées
    df = faostat.get_data_df('FS', pars=mypars, strval=False) # FS correspond au jeu d'indicateurs sur la sécurité alimentaire
    
    df = df.drop(['Domain Code', 'Domain', 'Area Code', 'Element Code', 'Element', 'Year Code','Item Code'], axis=1)
    df = df.rename(columns={
        'Item': 'Indicateur',
        'Area': 'Pays',
        'Year': 'Année',
        'Unit' : 'Unité',
        'Value' : 'Valeur'
        })

    for year in df['Année'].unique():
        # Pour l'instant la valeur de year s'écrit 'xxxx-yyyy'. Comme on ne veut garder que la première année ('xxxx'). 
        # On ne garde que les 4 premiers caractères. On les passe ensuite en entier.
        df.loc[df['Année'] == year, 'Année'] = int(year[:4])

    for indicateur in df['Indicateur']:
        if indicateur == 'Average dietary energy supply adequacy (percent) (3-year average)':
            df.loc[df['Indicateur'] == indicateur, 'Indicateur'] = "suffisance des apports énergétiques alimentaires moyens"

        elif indicateur == 'Dietary energy supply used in the estimation of the prevalence of undernourishment (kcal/cap/day) (3-year average)':
            df.loc[df['Indicateur'] == indicateur, 'Indicateur'] = "disponibilité alimentaire par habitant"

        elif indicateur == 'Average protein supply (g/cap/day) (3-year average)':
            df.loc[df['Indicateur'] == indicateur, 'Indicateur'] = "disponibilité protéiques moyenne"

        elif indicateur == 'Prevalence of undernourishment (percent) (3-year average)':
            df.loc[df['Indicateur'] == indicateur, 'Indicateur'] = "prévalence de la sous alimentation"
    return df




