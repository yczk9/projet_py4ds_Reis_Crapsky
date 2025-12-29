# Projet Python 

*Par Yann CRAPSY et Ophélie REIS, 2025.*

# Sommaire
1. [Définitions](#definitions)
2. [Démarche et objectifs](#objectifs)
3. [Sources des données](#sources)
4. [Présentation du dépôt](#pres)
5. [Licence](#licence)



## 1. Définitions <a name="definitions">

**Dette publique au sens de Maastricht :** 

La dette au sens de Maastricht, ou dette publique notifiée, couvre l'ensemble des administrations publiques au sens des comptes nationaux : l'État, les organismes divers d'administration centrale (ODAC), les administrations publiques locales et les administrations de sécurité sociale. 

(Source : https://www.insee.fr/fr/metadonnees/definition/c1091.)

**Indice de développement humain :**

Indice agrégé, moyenne géométrique de trois sous-indices relatifs respectivement à la santé, l'éducation et le revenu de la population. Compris entre 0 (exécrable) et 1 (excellent), les valeurs actuelles vont, en 2021, de 0,95+ (Europe du Nord) à 0,4- (certains pays d'Afrique), avec une médiane autour de 0,7.

## 2. Démarche et objectifs <a name="objectifs">

Le PIB est un indicateur parfois jugé obsolète en raison, en partie, de sa difficulté  à mesurer le bien-être. Cette difficulté s'accroit notamment à un seuil de développement à partir duquel on observe que le PIB devient complètement décorélé du bien-être. 
Nous avons voulu capturer cet effet de seuil au sujet de la sécurité alimentaire. L'objectif était de tenter de voir à partir de quand la hausse du PIB n'impliquait plus hausse de la sécurité alimentaire. 
Pour cela, nous avons choisi de travailer sur les BRICSAM (Brésil, Russie, Inde, Chine, Afrique du Sud, Mexique). La littérature économique sur le sujet suggère en effet que l'effet de seuil est observable au moment où un pays est considéré comme "développé". En choisissant ce groupe de pays en fin de transition nous espérions donc avoir le bon échantillon pour capturer cet effet de seuil.  

Pour créer une variable de sécurité alimentaire, nous avons réalisé une analyse en composante principale sur les trois variables suivantes: disponibilité alimentaire par habitant (kcal/cap/d)','disponibilité protéiques moyenne (g/cap/d)','suffisance des apports énergétiques alimentaires moyens (%)'. Celapermettait d'avoir un indictaur synthétique. Nous puvions finalement régresser le PIB sur ces indicateurs pour observer le force de la corrélation. Notre hypothèse était que nous pourrions observer un effet de seuil, ce qui ce serait traduit par un affaiblissement de la corrélation entre 2000 et 2020.    

## 3. Sources des données <a name="sources">

Nous avons utilisé les datasets:

- De la Banque mondiale pour le PIB
- De faostat pour les données de sécurité alimentaires. Ce jeu de données nous donnait accès à des variables de sécurité alimentaire de 2000 à 2020, ce qui semblait suffisament large pour mener notre étude.

## 4. Présentation du dépôt <a name=pres>

Notre production est essentiellement localisée dans deux versions d'un fichier ```main.ipynb```.
- La première ne contient que le code non exécuté et les commentaires entre les cellules. 
- Le code dans la seconde a été préalablement exécuté, afin de pouvoir présenter les résultats même en cas  d'inaccessibilité temporaire des sources. 

C'est cette version exécutée qui tient lieu de rapport final.

Le dossier ```data``` contient une copie locale d'une partie des données tirées de nos sources. Les API  correspondantes ont été indisponibles pendant quelques jours durant le projet, ce qui nous a contraint à trouver une parade.

Le dossier ```scripts``` contient, comme on l'imagine, une multitude de fonctions utiles, afin de rendre notre code plus lisible et maintenanble. 

Quant au fichier ```requirements.txt```, il est appelé par pip afin d'installer les paquets nécessaires en début d'exécution.

## 5. Licence <a name="licence">

Ce projet est sous licence GPLv3.

