import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def select_acp_columns(df):
    """
    Identifie et extrait les colonnes nécessaires pour l'ACP.
    """
    cols_dispo = df.columns.tolist()
    try:
        col_kcal = [c for c in cols_dispo if "kcal" in c and "disponibilité" in c][0]
        col_prot = [c for c in cols_dispo if "protéiques" in c or "protein" in c.lower()][0]
        col_suff = [c for c in cols_dispo if "suffisance" in c or "adequacy" in c.lower()][0]
        
        cols_acp = [col_kcal, col_prot, col_suff]
        print(f"Colonnes sélectionnées : {cols_acp}")
        return cols_acp
    except IndexError:
        print("ERREUR : Colonnes introuvables. Vérifiez les noms des variables.")
        return None

def preprocess_and_scale(df, cols_acp):
    """
    Nettoie les NaNs et standardise les données.
    """
    df_clean = df.dropna(subset=cols_acp).copy()
    X = df_clean[cols_acp].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df_clean, X_scaled

def run_pca_and_get_scores(df_clean, X_scaled, cols_acp):
    """
    Exécute l'ACP et oriente l'axe F1 pour qu'il représente la sécurité alimentaire.
    """
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    
    # Calcul de corrélation pour l'orientation de F1
    # On compare F1 avec la 1ère variable brute (souvent kcal)
    correlation_f1_var1 = np.corrcoef(coords[:, 0], df_clean[cols_acp[0]])[0, 1]
    
    # Correction de l'orientation si nécessaire
    if correlation_f1_var1 < 0:
        coords[:, 0] = -coords[:, 0]
        # On inverse aussi les composantes pour le graphique plus tard
        pca.components_[0, :] = -pca.components_[0, :]
        print("Axe F1 inversé pour aligner les valeurs positives avec une meilleure sécurité.")

    df_clean['Score_Securite_Alim_F1'] = coords[:, 0]
    df_clean['Score_Securite_Alim_F2'] = coords[:, 1]
    
    return df_clean, pca

def plot_correlation_circle(pca, cols_acp, filename="cercle_correlations.png"):
    """
    Génère et sauvegarde le cercle des corrélations.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--')
    ax.add_artist(circle)

    comps = pca.components_.T
    for i, (x, y) in enumerate(zip(comps[:, 0], comps[:, 1])):
        ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='b', ec='b')
        ax.text(x * 1.15, y * 1.15, cols_acp[i].split('(')[0], color='b', ha='center')

    ax.set_xlabel(f"F1 - Indice Synthétique ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"F2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Cercle des corrélations : Construction de l'Indice")
    ax.grid(linestyle='--')
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Graphique sauvegardé sous {filename}")