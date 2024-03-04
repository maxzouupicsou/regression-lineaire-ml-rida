# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# On charge le dataset
insurance_data = pd.read_csv('insurancev2.csv')  # Correction du chemin du fichier

# On affiche le nuage de points dont on dispose
plt.scatter(insurance_data['bmi'], insurance_data['charges'], color='blue', marker='o', s=10)  # Ajustement des colonnes et des paramètres de style
plt.title('Relation entre BMI et Charges d\'assurance')
plt.xlabel('BMI')
plt.ylabel('Charges d\'assurance')
plt.show()

# On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul
X = insurance_data[['bmi']].values  # Sélection correcte de la colonne BMI
Y = insurance_data['charges'].values  # Sélection correcte de la colonne Charges

# Création du modèle de régression linéaire
model = LinearRegression()
model.fit(X, Y)

# Calcul des prédictions pour tracer la ligne de régression
Y_pred = model.predict(X)

# Affichage des données et de la ligne de régression
plt.scatter(X, Y, color='blue', marker='o', s=10, label='Données réelles')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Ligne de régression')
plt.title('Régression Linéaire entre BMI et Charges d\'assurance')
plt.xlabel('BMI')
plt.ylabel('Charges d\'assurancnne')
plt.legend()
plt.show()
