# Description

Ce projet consiste en l'analyse et la prédiction des données du Titanic à partir d'un fichier CSV, en utilisant des techniques d'apprentissage automatique. Les deux algorithmes clés sont :

- script.py : Ce code implémente un modèle de régression logistique pour prédire la probabilité de survie d'une personne en fonction de son âge, de sa classe et de son nom. Le code utilise également LabelEncoder pour encoder les données catégorielles, train_test_split pour diviser les données en ensembles de formation et de test, et pandas pour charger les données depuis un fichier CSV.
- titanicKmean : Ce code utilise l'algorithme de clustering KMeans pour regrouper les instances du Titanic en 3 clusters, en utilisant les attributs Age, Headlen, Headwth, Neck, Length, Chest et Weight. Le code utilise également MinMaxScaler pour normaliser les données et PCA pour réduire la dimensionnalité des données. Les bibliothèques nécessaires pour exécuter ce code sont pandas, numpy, sklearn.cluster.KMeans, sklearn.preprocessing.MinMaxScaler, sklearn.decomposition.PCA et matplotlib.pyplot.

## Instructions

Pour exécuter le code, assurez-vous d'avoir les bibliothèques nécessaires installées sur votre machine. Pour cela, vous pouvez utiliser la commande pip install suivie du nom de la bibliothèque. Par exemple, pip install pandas.

Ensuite, exécutez le code à partir d'un IDE ou d'un terminal en utilisant la commande python nom_du_fichier.py. Avant d'exécuter le code, assurez-vous de spécifier les noms de fichiers appropriés pour charger les données Titanic.

Pour le script.py, il vous sera demandé d'entrer le nom, la classe et l'âge d'une personne pour laquelle vous souhaitez prédire la probabilité de survie.

Pour le titanicKmean.py , vous pouvez modifier les attributs à utiliser pour le clustering en modifiant la liste des attributs sélectionnés dans le code.

### Auteur

Ce projet a été créé par Julien Carrillo.