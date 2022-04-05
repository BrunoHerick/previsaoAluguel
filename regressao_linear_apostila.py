from random import random
import pandas as pd
import matplotlib.pyplot as grafico
import sklearn.linear_model as Linear
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# carregar os dados
dados = pd.read_csv("/home/bhs/PROFISSIONAL/PYTHON/SCIKIT_LEARN/previsao_renda_aluguel/previsao_aluguel.csv")

# criar indice
dados_com_indice = dados.reset_index()# criar coluna indice
dados_com_indice["index"] = dados_com_indice["index"]*10

# criar testes
paraTreino, paraTeste = train_test_split(dados_com_indice, test_size=0.2, random_state=42)

paraTreinoRenda = np.array(paraTreino.iloc[:,1]).reshape(-1,1)
paraTreinoAluguel = np.array(paraTreino.iloc[:,2]).reshape(-1,1)

paraTesteRenda = np.array(paraTeste.iloc[:,1]).reshape(-1,1)
paraTesteAluguel = np.array(paraTeste.iloc[:,2]).reshape(-1,1)

# criar modelo
modelo = Linear.LinearRegression()

# treinar modelo
modelo.fit(paraTreinoRenda,paraTreinoAluguel)

# fazer previsao para Treino
novaRenda = [[3000]]
print("Treino",modelo.predict(novaRenda))

# valor de a e b
print("y = {} + {}x".format(modelo.intercept_, modelo.coef_))

# treinar modelo para teste
modelo.fit(paraTesteRenda,paraTesteAluguel)

# fazer previsao para Teste
novaRenda = [[3000]]
print("Teste",modelo.predict(novaRenda))

# valor de a e b
print("y = {} + {}x".format(modelo.intercept_, modelo.coef_))

# r pearson
matris_corelacao = dados.corr()
print(matris_corelacao["renda"].sort_values(ascending=False))

# erro medio
raiz_erro_medio = mean_squared_error(paraTreinoRenda, paraTreinoAluguel)
print(np.sqrt(raiz_erro_medio))

print(np.sqrt(mean_squared_error(dados_com_indice.iloc[:,1], dados_com_indice.iloc[:,2])))