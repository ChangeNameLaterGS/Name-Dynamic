import numpy as np
import matplotlib.pyplot as plt

# Função para calcular a distância euclidiana entre dois pontos
def distancia_euclidiana(ponto1, ponto2):
  """Calcula a distância euclidiana entre dois pontos.

  Args:
    ponto1: Um array numpy de duas dimensões representando as coordenadas do primeiro ponto.
    ponto2: Um array numpy de duas dimensões representando as coordenadas do segundo ponto.

  Returns:
    A distância euclidiana entre os dois pontos.
  """
  return np.sqrt(np.sum((ponto1 - ponto2)**2))

# Função para encontrar a distribuição ótima de sensores
def encontrar_distribuicao_otima(pontos_de_interesse, num_sensores):
  """Encontra a distribuição ótima de sensores utilizando programação dinâmica.

  Args:
    pontos_de_interesse: Um array numpy de duas dimensões representando as coordenadas dos pontos de interesse.
    num_sensores: O número de sensores disponíveis para implantação.

  Returns:
    Um array numpy de duas dimensões representando as coordenadas dos sensores na distribuição ótima.
  """
  # Inicializa a matriz de custos
  custos = np.zeros((num_sensores, len(pontos_de_interesse)))
  # Inicializa a matriz de predecessores
  predecessores = np.zeros((num_sensores, len(pontos_de_interesse)), dtype=int)

  # Calcula os custos para o primeiro sensor
  for j in range(len(pontos_de_interesse)):
    custos[0, j] = distancia_euclidiana(pontos_de_interesse[j], pontos_de_interesse[0])
    predecessores[0, j] = 0

  # Calcula os custos para os sensores restantes
  for i in range(1, num_sensores):
    for j in range(len(pontos_de_interesse)):
      # Encontra o ponto de interesse mais próximo ao sensor atual
      minimo = float('inf')
      indice_minimo = -1
      for k in range(len(pontos_de_interesse)):
        custo_atual = custos[i-1, k] + distancia_euclidiana(pontos_de_interesse[j], pontos_de_interesse[k])
        if custo_atual < minimo:
          minimo = custo_atual
          indice_minimo = k
      custos[i, j] = minimo
      predecessores[i, j] = indice_minimo

  # Encontra a distribuição ótima
  distribuicao_otima = np.zeros((num_sensores, 2))
  indice_atual = np.argmin(custos[num_sensores-1, :])
  distribuicao_otima[num_sensores-1] = pontos_de_interesse[indice_atual]

  for i in range(num_sensores-2, -1, -1):
    indice_atual = predecessores[i+1, indice_atual]
    distribuicao_otima[i] = pontos_de_interesse[indice_atual]

  return distribuicao_otima

# Define os pontos de interesse
pontos_de_interesse = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

# Define o número de sensores
num_sensores = 3

# Encontra a distribuição ótima de sensores
distribuicao_otima = encontrar_distribuicao_otima(pontos_de_interesse, num_sensores)

# Plota os pontos de interesse e a distribuição ótima de sensores
plt.figure(figsize=(8, 8))
plt.scatter(pontos_de_interesse[:, 0], pontos_de_interesse[:, 1], s=50, c='blue', label='Pontos de Interesse')
plt.scatter(distribuicao_otima[:, 0], distribuicao_otima[:, 1], s=100, c='red', marker='*', label='Sensores')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Distribuição Ótima de Sensores')
plt.legend()
plt.show()