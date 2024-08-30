# Importa as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from VECM import VECM_Model

# Carrega o arquivo Excel e seleciona as colunas necessárias
data = pd.read_excel("dados.xlsx")
data.set_index('Ano-Mês', inplace=True)
data_set = data[["Crédito", "CDI a.m", "IPCA a.m"]]

# Escala os dados
sc = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = sc.fit_transform(data_set)

# Prepara os dados para o modelo LSTM
X = []
y = []
backcandles = 6

for i in range(backcandles, len(data_set_scaled)):
    X.append(data_set_scaled[i-backcandles:i, :])
    y.append(data_set_scaled[i, 0])

X, y = np.array(X), np.array(y)

# Separa os dados de treinamento e teste
splitlimit = int(len(X) * 0.8)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]

# Cria o modelo
model = Sequential()
model.add(LSTM(units=150, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=150, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear'))

# Compila o modelo
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Treina o modelo
model.fit(X_train, y_train, epochs=30, batch_size=15, validation_split=0.1, shuffle=True)

# Faz previsões para todos os dados disponíveis
y_pred = model.predict(X)

# Previsão para 5 períodos à frente
for _ in range(5):
    last_sequence = X[-1][1:]  # Pega as últimas observações, remove a mais antiga
    last_pred = np.array([y_pred[-1]])  # Converte a previsão em array de dimensão correta
    last_pred = np.tile(last_pred, (1, X.shape[2]))  # Expande a previsão para ter o mesmo número de features
    last_sequence = np.vstack([last_sequence, last_pred])  # Adiciona a nova previsão ao final
    new_input = last_sequence.reshape((1, backcandles, X.shape[2]))  # Ajusta a forma para ser compatível
    new_pred = model.predict(new_input)  # Faz a nova previsão
    y_pred = np.append(y_pred, new_pred, axis=0)  # Adiciona a nova previsão à lista de previsões

# Obtenha o número de features
num_features = data_set.shape[1]

# Desfaz a escala das previsões
y_pred_extended = np.concatenate((y_pred, np.zeros((y_pred.shape[0], num_features - 1))), axis=1)
y_pred = sc.inverse_transform(y_pred_extended)[:, 0]

y_test_extended = np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], num_features - 1))), axis=1)
y_test = sc.inverse_transform(y_test_extended)[:, 0]

# Cria novas datas para os períodos à frente
dates = data.index.to_list()
future_dates = pd.date_range(start=dates[-1], periods=6, freq='M')[1:]
dates.extend(future_dates)

# Ajusta o comprimento das previsões e das datas
if len(dates) > len(y_pred):
    dates = dates[:len(y_pred)]  # Ajusta 'dates' para o comprimento de 'y_pred'
elif len(y_pred) > len(dates):
    y_pred = y_pred[:len(dates)]  # Ajusta 'y_pred' para o comprimento de 'dates'

# Converte as datas para strings no formato adequado
dates_str = [date.strftime('%Y-%m') if isinstance(date, pd.Timestamp) else date for date in dates]

# Salva as previsões finais no DataFrame
df_result = pd.DataFrame({
    'Data': dates[:len(y_pred)],  # Ajusta 'Data' para o comprimento de 'y_pred'
    'Crédito Previsto': y_pred
})

df_result.set_index('Data', inplace=True)
df_result.to_excel('final_todos_periodos.xlsx')

# Análise com o VECM Model
df_result_set = pd.concat([df_result, data_set[["CDI a.m", "IPCA a.m"]][:len(df_result)]], axis=1).dropna()

modelo_vecm = VECM_Model(df_result_set, target="Crédito Previsto", index=0, diff=5, coint=1, deterministic="li")
modelo_vecm.fit_model()
dados_finais = modelo_vecm.predict_model(pfrente=5)

# Salva os resultados das previsões do VECM
dados_finais.to_excel('previsoes_vecm_final.xlsx')