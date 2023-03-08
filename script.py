import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
# Charger les données Titanic
df = pd.read_csv("titanic.csv")

# Encoder les données catégorielles (ici la classe sociale)
le = LabelEncoder()
df["Pclass"] = le.fit_transform(df["Pclass"])


# Sélectionner les colonnes d'intérêt
X = df[["Age", "Pclass"]]
y = df["Survived"]

# Supprimer les lignes avec des valeurs manquantes
X = X.dropna()
y = y[X.index]

# Entraîner un modèle de régression logistique
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=10000).fit(X_train, y_train)

# Demander à l'utilisateur de saisir les informations sur la personne
name = input("Nom de la personne : ")
pclass = input("Classe (1,2,3) : ")
age = int(input("Age : "))

# Convertir la classe sociale en un nombre (0 pour la 1ère classe, 1 pour la 2ème classe, etc.)
pclass = le.transform([pclass])[0]

# Prédire la probabilité de survie de la personne
X_new = np.array([[age, pclass]])
prob = model.predict_proba(X_new)

# Afficher le résultat
print(f"{name} a une probabilité de {prob[0][0]:.6f} de survivre")
print("Recap: ")
print(f"Age: {age}")
print(f"Class : {pclass + 1}")

# Importer les bibliothèques pandas, numpy, LogisticRegression, LabelEncoder et train_test_split
# Ignorer les avertissements lors de l'exécution du code
# Charger les données Titanic depuis un fichier CSV et stocker les données dans un DataFrame
# Encoder les données catégorielles (ici la classe sociale) en utilisant LabelEncoder
# Sélectionner les colonnes d'intérêt, y compris l'âge et la classe sociale, et stocker les données dans des variables distinctes
# Supprimer les lignes avec des valeurs manquantes
# Diviser les données en ensembles de formation et de test à l'aide de train_test_split, en spécifiant une taille de test de 0,2 et une graine aléatoire de 0
# Entraîner un modèle de régression logistique sur les données d'entraînement
# Demander à l'utilisateur d'entrer les informations sur la personne pour laquelle nous souhaitons prédire la probabilité de survie, telles que le nom, la classe et l'âge
# Convertir la classe sociale en un nombre (0 pour la 1ère classe, 1 pour la 2ème classe, etc.) en utilisant LabelEncoder
# Créer un tableau avec les valeurs d'âge et de classe sociale de la personne pour laquelle nous souhaitons prédire la probabilité de survie
# Prédire la probabilité de survie de la personne en utilisant le modèle entraîné
# Afficher le résultat, y compris le nom de la personne et sa probabilité de survie, ainsi qu'un récapitulatif de l'âge et de la classe sociale de la personne.