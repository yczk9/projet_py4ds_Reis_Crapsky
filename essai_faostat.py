import pandas as pd

fichier ="C:/Users/laeti/OneDrive/Bureau/FAOSTAT_data_filtre2.csv"

df = pd.read_csv(fichier, encoding="utf-8") 

df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Zone": "country",
    "Année": "year",
    "Code Produit": "variable_code",
    "Valeur": "value"
})

# --- Garder uniquement les codes produits numériques ---
df = df[df["variable_code"].astype(str).str.isnumeric()]
df["variable_code"] = df["variable_code"].astype(int)

# --- Convertir les valeurs en float ---
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# --- Codes produits d'intérêt ---
variables_dict = {
    21010: "suffisance des apports énergétiques alimentaires moyens",
    22000: "disponibilité alimentaire par habitant",
    21013: "disponibilité protéiques moyenne",
    210104: "disponibilité protéines moyennes animales",
    22013: "PIB/ HAB",
    210041: "prévalence de la sous alimentation",
    210401: "prévalence de l'insécurité alimentaire grave",
    210091: "prévalence de l'insécurité alimentaire modérée ou grave",
    21031: "variabilité des disponibilités alimentaires"
}

variables = list(variables_dict.keys())

df_filtre = df[df["variable_code"].isin(variables)]

# --- Pivot : country × year avec une colonne par variable ---
df_wide = df_filtre.pivot_table(
    index=["country", "year"],
    columns="variable_code",
    values="value"
).reset_index()

# --- Ne garder que les variables présentes après pivot ---
variables_presentes = [v for v in variables if v in df_wide.columns]

# --- Statistiques globales ---
stats_base = df_wide[variables_presentes].agg(
    ["mean", "var", "std", "min", "max", "median"]
).T

# --- Déciles D10 à D90 ---
deciles = df_wide[variables_presentes].quantile(
    [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
).T
deciles.columns = [f"D{int(q*10)}" for q in deciles.columns]

# --- Fusion des stats et déciles ---
stats = pd.concat([stats_base, deciles], axis=1)

# --- Ajouter la colonne "Variable" ---
stats.insert(0, "Variable", [variables_dict[v] for v in stats.index])

print("=== Statistiques globales ===")
print(stats)
print("\n")

pays_selectionnes = [ "France", "Royaume-Uni", "États-Unis d'Amérique", "Fédération de Russie", "Chine", "Brésil", "Afrique du Sud", "Japon" ]
df_pays = df_wide[df_wide["country"].isin(pays_selectionnes)]
colonnes_a_afficher = ["country", "year"] + variables_presentes
print("=== Valeurs pour les pays sélectionnés ===")
print(df_pays[colonnes_a_afficher])


# import tkinter as tk
# from tkinter import ttk

# def afficher_table(df, titre="Tableau"):
#     root = tk.Tk()
#     root.title(titre)

#     frame = ttk.Frame(root)
#     frame.pack(fill='both', expand=True)

#     tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
#     tree.pack(fill='both', expand=True)

#     for col in df.columns:
#         tree.heading(col, text=col)
#         tree.column(col, width=100)

#     for _, row in df.iterrows():
#         tree.insert("", "end", values=list(row))

#     root.mainloop()

# afficher_table(stats, titre="Statistiques globales")
# afficher_table(df_pays[colonnes_a_afficher], titre="Pays sélectionnés")

# Exporter les statistiques globales
stats.to_excel("stats_globales.xlsx", index=False)

# Exporter les données des pays sélectionnés
df_pays[colonnes_a_afficher].to_excel("donnees_pays_selectionnes.xlsx", index=False)
