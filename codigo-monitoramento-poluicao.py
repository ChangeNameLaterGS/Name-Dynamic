import numpy as np
import matplotlib.pyplot as plt

# Define a função para calcular a distância entre dois pontos
def distancia_euclidiana(ponto1, ponto2):
  """
  Calcula a distância euclidiana entre dois pontos.

  Args:
    ponto1: Um array numpy de duas dimensões representando o primeiro ponto (x, y).
    ponto2: Um array numpy de duas dimensões representando o segundo ponto (x, y).

  Returns:
    A distância euclidiana entre os dois pontos.
  """
  return np.sqrt(np.sum((ponto1 - ponto2)**2))

# Define a função para calcular a distância considerando a corrente marítima
def distancia_com_corrente(ponto1, ponto2, corrente):
  """
  Calcula a distância entre dois pontos considerando a influência da corrente marítima.

  Args:
    ponto1: Um array numpy de duas dimensões representando o primeiro ponto (x, y).
    ponto2: Um array numpy de duas dimensões representando o segundo ponto (x, y).
    corrente: Um array numpy de duas dimensões representando a direção e velocidade da corrente (x, y).

  Returns:
    A distância entre os dois pontos considerando a corrente.
  """
  # Calcula a distância euclidiana
  distancia_base = distancia_euclidiana(ponto1, ponto2)

  # Calcula a influência da corrente
  influencia_corrente = np.dot(ponto2 - ponto1, corrente)

  # Retorna a distância ajustada pela corrente
  return distancia_base + influencia_corrente

# Define a função para encontrar a distribuição ótima dos sensores
def distribuicao_otima_sensores(pontos, num_sensores, corrente):
  """
  Encontra a distribuição ótima dos sensores usando programação dinâmica, considerando a corrente marítima.

  Args:
    pontos: Um array numpy de duas dimensões representando as coordenadas dos pontos de interesse (x, y).
    num_sensores: O número de sensores a serem distribuídos.
    corrente: Um array numpy de duas dimensões representando a direção e velocidade da corrente (x, y).

  Returns:
    Um array numpy de duas dimensões representando as coordenadas dos sensores otimizados.
  """
  # Inicializa a tabela de programação dinâmica
  tabela_dp = np.zeros((num_sensores + 1, len(pontos)))

  # Inicializa a tabela de predecessores
  predecessores = np.zeros((num_sensores + 1, len(pontos)), dtype=int)

  # Preenche a primeira linha da tabela
  for i in range(len(pontos)):
    tabela_dp[0, i] = distancia_com_corrente(pontos[0], pontos[i], corrente)

  # Preenche as linhas restantes da tabela
  for i in range(1, num_sensores + 1):
    for j in range(len(pontos)):
      # Encontra o ponto de interesse anterior com a menor distância acumulada
      min_distancia = float('inf')
      min_index = -1
      for k in range(j):
        distancia_acumulada = tabela_dp[i - 1, k] + distancia_com_corrente(pontos[k], pontos[j], corrente)
        if distancia_acumulada < min_distancia:
          min_distancia = distancia_acumulada
          min_index = k
      # Atualiza a tabela com a menor distância acumulada
      tabela_dp[i, j] = min_distancia
      # Salva o predecessor
      predecessores[i, j] = min_index

  # Encontra o último sensor
  ultimo_sensor = np.argmin(tabela_dp[num_sensores])

  # Reconstrói o caminho a partir do último sensor
  sensores_otimos = [pontos[ultimo_sensor]]
  for i in range(num_sensores - 1, 0, -1):
    ultimo_sensor = predecessores[i, ultimo_sensor]
    sensores_otimos.append(pontos[ultimo_sensor])

  # Retorna a distribuição ótima dos sensores
  return np.array(sensores_otimos[::-1])

# Exemplo de uso
# Define as coordenadas dos pontos de interesse
pontos = np.array([[1, 1], [2, 3], [4, 2], [5, 5], [3, 4]])

# Define o número de sensores
num_sensores = 3

# Define a direção e velocidade da corrente
corrente = np.array([0.5, 0.2])  # Velocidade de 0.5 unidades na direção x e 0.2 unidades na direção y

# Encontra a distribuição ótima dos sensores
sensores_otimos = distribuicao_otima_sensores(pontos, num_sensores, corrente)

# Plota os pontos e os sensores otimizados
plt.plot(pontos[:, 0], pontos[:, 1], 'bo')
plt.plot(sensores_otimos[:, 0], sensores_otimos[:, 1], 'rX')
plt.title('Distribuição Ótima de Sensores (considerando a corrente)')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.show()