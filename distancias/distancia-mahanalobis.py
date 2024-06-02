import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

# Define a função para calcular a distância de Mahalanobis entre dois pontos
def distancia_mahalanobis(x, y, cov):
  """
  Calcula a distância de Mahalanobis entre dois pontos.

  Args:
    x: O primeiro ponto.
    y: O segundo ponto.
    cov: A matriz de covariância.

  Returns:
    A distância de Mahalanobis entre os dois pontos.
  """
  return mahalanobis(x, y, cov)

# Define a função para encontrar a distribuição ótima dos sensores
def encontrar_distribuicao_otima(pontos, num_sensores, cov):
  """
  Encontra a distribuição ótima dos sensores usando programação dinâmica.

  Args:
    pontos: Um array de pontos (coordenadas) onde os sensores podem ser colocados.
    num_sensores: O número de sensores a serem colocados.
    cov: A matriz de covariância.

  Returns:
    Um array com as coordenadas dos sensores otimizados.
  """
  # Cria uma matriz para armazenar os custos mínimos
  custos = np.zeros((num_sensores + 1, len(pontos)))
  # Cria uma matriz para armazenar a posição do sensor anterior
  predecessores = np.zeros((num_sensores + 1, len(pontos)), dtype=int)

  # Inicializa a primeira linha da matriz de custos
  for j in range(len(pontos)):
    custos[0, j] = 0

  # Itera sobre os sensores
  for i in range(1, num_sensores + 1):
    # Itera sobre os pontos
    for j in range(len(pontos)):
      # Encontra o custo mínimo para colocar o sensor no ponto atual
      custo_minimo = float('inf')
      predecessor_minimo = -1
      # Itera sobre os pontos anteriores
      for k in range(j):
        # Calcula a distância de Mahalanobis entre os pontos
        distancia = distancia_mahalanobis(pontos[j], pontos[k], cov)
        # Calcula o custo de colocar o sensor no ponto atual
        custo = custos[i - 1, k] + distancia
        # Se o custo atual for menor que o custo mínimo atual, atualiza o custo mínimo e o predecessor
        if custo < custo_minimo:
          custo_minimo = custo
          predecessor_minimo = k
      # Atualiza a matriz de custos e a matriz de predecessores
      custos[i, j] = custo_minimo
      predecessores[i, j] = predecessor_minimo

  # Encontra a posição do último sensor
  ultimo_sensor = np.argmin(custos[num_sensores])
  # Cria um array para armazenar as coordenadas dos sensores
  sensores = np.zeros((num_sensores, 2))

  # Reconstrói a sequência de sensores a partir da matriz de predecessores
  for i in range(num_sensores - 1, -1, -1):
    sensores[i] = pontos[ultimo_sensor]
    ultimo_sensor = predecessores[i + 1, ultimo_sensor]

  # Retorna as coordenadas dos sensores otimizados
  return sensores

# Define os pontos (coordenadas) onde os sensores podem ser colocados
pontos = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
# Define o número de sensores
num_sensores = 3
# Define a matriz de covariância
cov = np.array([[1, 0], [0, 1]])

# Encontra a distribuição ótima dos sensores
sensores = encontrar_distribuicao_otima(pontos, num_sensores, cov)

# Plota os pontos e os sensores
plt.scatter(pontos[:, 0], pontos[:, 1], label='Pontos')
plt.scatter(sensores[:, 0], sensores[:, 1], label='Sensores')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Distribuição Ótima de Sensores')
plt.legend()
plt.show()