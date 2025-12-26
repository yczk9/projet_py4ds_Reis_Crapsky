import pandas as pd

def parse_worldbank_pib(data_json):
    """
    Transforme le JSON brut de la Banque Mondiale en DataFrame structuré.
    """
    list = []
    for item in data_json:
        list.append({
            'Pays': item['country']['value'],
            'Année': int(item['date']),
            'Valeur': item['value']
        })
    return pd.DataFrame(list)

def filter_years(df, start_year, end_year):
    """
    Filtre le DataFrame sur une plage d'années spécifique.
    """
    mask = (df['Année'] >= start_year) & (df['Année'] <= end_year)
    return df[mask].copy()

def filter_bricsam_countries(df,pays_a_conserver):
    """
    Filtre le DataFrame pour ne garder que les pays sélectionnés.
    """
    return df[df['Pays'].isin(pays_a_conserver)].copy()

