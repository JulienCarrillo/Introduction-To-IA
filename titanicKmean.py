import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Importer la base de données titanic
titanic_df = pd.read_csv('titanic.csv', delimiter=',', header=0, encoding='utf-8-sig')

# Sélectionner les attributs
X = titanic_df.iloc[:, [False,True,True,False,False,True,False,False,False,False,False,False]].values

# Normaliser les attributs
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Utiliser l'algorithme de clustering KMeans pour regrouper les instances en 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# Réduire la dimensionnalité des données en utilisant PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Afficher les clusters dans un graphique 2D
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusters des différentes classes")
plt.show()

#algo 
# Importer les bibliothèques nécessaires : pandas, numpy, sklearn.cluster.KMeans, sklearn.preprocessing.MinMaxScaler, sklearn.decomposition.PCA et matplotlib.pyplot.
# Importer la base de données titanic à partir d'un fichier CSV en utilisant pandas.
# Sélectionner les attributs Age, Headlen, Headwth, Neck, Length, Chest et Weight.
# Normaliser les attributs en utilisant sklearn.preprocessing.MinMaxScaler.
# Utiliser l'algorithme de clustering KMeans pour regrouper les instances en 3 clusters en utilisant sklearn.cluster.KMeans.
# Réduire la dimensionnalité des données en utilisant PCA en utilisant sklearn.decomposition.PCA.
# Afficher les clusters dans un graphique 2D en utilisant matplotlib.pyplot.