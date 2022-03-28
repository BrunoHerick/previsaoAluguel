import pandas as pd
import matplotlib.pyplot as grafico
from sklearn.linear_model import LinearRegression
import numpy as np

renda = np.array([5009.84,7230.56,5288.76,3568.04,3905.72,5231.13,5464.76,1121.89,1017.32,961.66,956.82,3512.23,3161.76,2480.28])
aluguel = [1658.78,2234.1,1504.91,1401.3,1095.04,1748.39,3009.75,392.78,274.17,228.33,303.33,1254.49,1211.21,672.37]
renda = renda.reshape(-1,1)

modelo = LinearRegression()
modelo.fit(renda,aluguel)# treinar

coeficiente_angular_b = modelo.coef_
coeficiente_linear_a = modelo.intercept_

# y = coeficiente_linear_a + (coeficiente_angular_b * renda)
# print(coeficiente_angular_b, coeficiente_linear_a) --> (0.36, -60.427)

grafico.scatter(renda, aluguel)
grafico.scatter(3000, coeficiente_linear_a + (coeficiente_angular_b * 3000), color="green")# preco do aluguel para uma renda de 3000
grafico.grid()
grafico.show()