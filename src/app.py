import pandas as pd
import numpy as numpy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

url = 'https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv'
df = pd.read_csv(url)

# Definir variable target
target = 'ICU Beds_x'

# Regularización Lasso
X = df.drop(columns=[target])
y = df[target]

X = X.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)

pipeline = make_pipeline(StandardScaler(), Lasso(alpha=3))
pipeline.fit(X_train, y_train)

lista_coeficientes = pipeline[1].coef_

lista_indices = []

for i in range(len(lista_coeficientes)):
    if lista_coeficientes[i]!=0:
        lista_indices.append(i)

lista_columnas = list(df.columns[lista_indices])

# Construcción del modelo
X_reg = X[lista_columnas]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y, test_size=0.3, random_state=37)

lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)

filename = 'regularized_lineal_regression_model.sav'
pickle.dump(lin_reg, open(filename, 'wb'))

y_train_pred = lin_reg.predict(X_train_reg)
y_test_pred = lin_reg.predict(X_test_reg)

RMSE_train = mean_squared_error(y_train_reg, y_train_pred, squared=False)
RMSE_test = mean_squared_error(y_test_reg, y_test_pred, squared=False)

print(F'RMSE train: {round(RMSE_train)}')
print(F'RMSE test: {round(RMSE_test)}')
print()
print('Se guardó el modelo.')