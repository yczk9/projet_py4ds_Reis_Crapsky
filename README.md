# Projet Python 

*Par Yann CRAPSY et Ophélie REIS, 2025.*

Le PIB est-il un indicateur robuste de la sécurité alimentaire? Etude sur les BRICSAM entre 2000 et 2020. 

# Sommaire
1. [Définitions](#definitions)
2. [Démarche et objectifs](#objectifs)
3. [Sources des données](#sources)
4. [Présentation du dépôt](#pres)
5. [Licence](#licence)



## 1. Définitions <a name="definitions">

**Sécurité alimentaire :** 

La banque mondiale définit la sécurité alimentaire comme suit: "la sécurité alimentaire existe lorsque tous les êtres humains ont, à tout moment, un accès physique et économique à une nourriture suffisante, saine et nutritive leur permettant de satisfaire leurs besoins énergétiques et leurs préférences alimentaires pour mener une vie saine et active ».

(Source :[https://www.banquemondiale.org/fr/topic/agriculture/brief/food-security-update/what-is-food-security#:~:text=Selon%20la%20d%C3%A9finition%20qui%20en,satisfaire%20leurs%20besoins%20%C3%A9nerg%C3%A9tiques%20et].)

**Produit Intérieur Brut :**

Le produit intérieur brut aux prix du marché vise à mesurer la richesse créée par tous les agents, privés et publics, sur un territoire national pendant une période donnée. Agrégat clé de la comptabilité nationale, il repré­sente le résultat final de l’activité de production des unités productrices résidentes.

(Source: [https://www.insee.fr/fr/metadonnees/definition/c1365] )

## 2. Démarche et objectifs <a name="objectifs">

Le PIB est un indicateur parfois jugé obsolète en raison, en partie, de sa difficulté  à mesurer le bien-être. Cette difficulté s'accroit notamment à un seuil de développement à partir duquel on observe que le PIB devient complètement décorélé du bien-être. 
Nous avons voulu capturer cet effet de seuil au sujet de la sécurité alimentaire. L'objectif était de tenter de voir à partir de quand la hausse du PIB n'impliquait plus hausse de la sécurité alimentaire. 
Pour cela, nous avons choisi de travailer sur les BRICSAM (Brésil, Russie, Inde, Chine, Afrique du Sud, Mexique). La littérature économique sur le sujet suggère en effet que l'effet de seuil est observable au moment où un pays est considéré comme "développé". En choisissant ce groupe de pays en fin de transition nous espérions donc avoir le bon échantillon pour capturer cet effet de seuil.  

Nous proposons en premier de visualiser les évolutions du PIB et nos différents indicateurs sur les 20 dernières années afin d'avoir une première intution du résultat avant d'implémenter nos méthodes statistiques. 

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


