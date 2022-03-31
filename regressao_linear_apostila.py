import pandas as pd
import matplotlib.pyplot as grafico
import sklearn.linear_model as Linear
import numpy as np

# carregar os dados
dados = pd.read_csv("/home/bhs/PROFISSIONAL/PYTHON/SCIKIT_LEARN/previsao_renda_aluguel/previsao_aluguel.csv")
renda = np.array(dados.iloc[:,0]).reshape(-1,1)
aluguel = np.array(dados.iloc[:,1]).reshape(-1,1)

# criar modelo
modelo = Linear.LinearRegression()

# treinar modelo
modelo.fit(renda,aluguel)

# fazer previsao
novaRenda = [[3000]]
print(modelo.predict(novaRenda))

# valor de a e b
print("B: {}".format(modelo.coef_))
print("A: {}".format(modelo.intercept_))