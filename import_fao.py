import faostat

def fetch_fao_security_data(area_codes, item_codes, years):
    """
    Extrait les données de sécurité alimentaire (FS) depuis l'API FAOSTAT.
    """
    params = {
        'area': area_codes,
        'element': [6120],  # Valeur uniquement (exclut l'intervalle de confiance)
        'item': item_codes,
        'year': years
    }
    # Récupération via le domaine 'FS' (Food Security)
    return faostat.get_data_df('FS', pars=params, strval=False)

def clean_fao_columns(df):
    """
    Supprime les colonnes techniques et renomme les axes principaux en français.
    """
    cols_to_drop = ['Domain Code', 'Domain', 'Area Code', 'Element Code', 
                    'Element', 'Year Code', 'Item Code']
    
    # On utilise errors='ignore' pour éviter les plantages si une colonne manque
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    rename_map = {
        'Item': 'Indicateur',
        'Area': 'Pays',
        'Year': 'Année',
        'Unit': 'Unité',
        'Value': 'Valeur'
    }
    return df.rename(columns=rename_map)

def translate_indicators(df):
    """
    Traduit les intitulés longs des indicateurs de sécurité alimentaire en français.
    """
    translate_map = {
        'Average dietary energy supply adequacy (percent) (3-year average)': 
            "suffisance des apports énergétiques alimentaires moyens",
        'Dietary energy supply used in the estimation of the prevalence of undernourishment (kcal/cap/day) (3-year average)': 
            "disponibilité alimentaire par habitant",
        'Average protein supply (g/cap/day) (3-year average)': 
            "disponibilité protéiques moyenne"
    }
    df['Indicateur'] = df['Indicateur'].replace(translate_map)
    return df

def convert_fao_years(df):
    """
    Convertit les périodes '2018-2020' en année de début '2018'.
    """
    # Utilisation de .str[:4] pour ne garder que la première année
    df['Année'] = df['Année'].astype(str).str[:4].astype(int)
    return df