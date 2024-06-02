import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist  # Para calcular a distância entre pontos

# Define a área de interesse (um retângulo)
area_largura = 100  # km
area_altura = 50  # km

# Define o número de sensores
num_sensores = 10

# Define a posição da fonte de poluição (um ponto no meio da área)
fonte_x = area_largura / 2
fonte_y = area_altura / 2

# Define o modelo de dispersão (neste caso, um modelo simples de dispersão radial)
def modelo_dispersao(distancia):
    """
    Calcula a concentração de poluição em um determinado ponto com base na distância da fonte.

    Args:
        distancia: A distância da fonte (em km).

    Returns:
        A concentração de poluição no ponto dado.
    """
    # Define um modelo simples de dispersão, onde a concentração diminui com o quadrado da distância
    return 1 / (distancia**2 + 1)

# Define a função para calcular a distância entre dois pontos
def distancia(ponto1, ponto2):
    """
    Calcula a distância euclidiana entre dois pontos.

    Args:
        ponto1: Um ponto (x, y)
        ponto2: Outro ponto (x, y)

    Returns:
        A distância euclidiana entre os dois pontos.
    """
    return np.sqrt((ponto1[0] - ponto2[0])**2 + (ponto1[1] - ponto2[1])**2)

# Define a função para encontrar a distribuição ótima dos sensores
def encontrar_distribuicao_otima(num_sensores, area_largura, area_altura, fonte_x, fonte_y, modelo_dispersao):
    """
    Usa programação dinâmica para encontrar a distribuição ótima dos sensores para maximizar a cobertura da área.

    Args:
        num_sensores: O número de sensores a serem alocados.
        area_largura: A largura da área.
        area_altura: A altura da área.
        fonte_x: A coordenada x da fonte de poluição.
        fonte_y: A coordenada y da fonte de poluição.
        modelo_dispersao: A função que calcula a concentração de poluição com base na distância da fonte.

    Returns:
        Uma lista de coordenadas de sensores otimizadas.
    """

    # Cria uma grade de pontos de grade para representar a área
    pontos_grade_x = np.linspace(0, area_largura, 20)  # Ajuste a resolução da grade conforme necessário
    pontos_grade_y = np.linspace(0, area_altura, 20)  # Ajuste a resolução da grade conforme necessário
    pontos_grade = np.array(np.meshgrid(pontos_grade_x, pontos_grade_y)).T.reshape(-1, 2)

    # Calcula as distâncias entre a fonte de poluição e todos os pontos da grade
    distancias_fonte = cdist([[fonte_x, fonte_y]], pontos_grade, metric='euclidean')

    # Calcula as concentrações de poluição em todos os pontos da grade
    concentracoes = modelo_dispersao(distancias_fonte).flatten()

    # Cria uma matriz para armazenar os valores de cobertura (o valor máximo de concentração
    # que pode ser detectado por um sensor em um ponto da grade)
    cobertura = np.zeros((len(pontos_grade), num_sensores + 1))

    # Inicializa a primeira coluna da matriz de cobertura com 0 (sem sensores)
    cobertura[:, 0] = 0

    # Cria uma lista para armazenar as posições dos sensores otimizadas
    posicoes_sensores_otimas = []

    # Loop para cada sensor
    for i in range(1, num_sensores + 1):
        # Loop para cada ponto da grade
        for j in range(len(pontos_grade)):
            # Calcula a concentração de poluição no ponto atual
            concentracao_atual = concentracoes[j]
            
            # Calcula a cobertura máxima para o ponto atual, considerando todos os sensores anteriores
            cobertura_maxima = 0
            for k in range(j):
                cobertura_maxima = max(cobertura_maxima, cobertura[k, i - 1] + concentracoes[k])

            # Define a cobertura do ponto atual com o sensor atual
            cobertura[j, i] = max(cobertura[j, i - 1], cobertura_maxima)

            # Se o valor de cobertura for maior que o valor de cobertura anterior, atualiza a posição do sensor
            if cobertura[j, i] > cobertura[j, i - 1] and len(posicoes_sensores_otimas) < i:
                if len(posicoes_sensores_otimas) == i - 1:
                    posicoes_sensores_otimas.append(pontos_grade[j])
                else:
                    posicoes_sensores_otimas[i - 1] = pontos_grade[j]

    # Retorna a lista de coordenadas dos sensores otimizadas
    return np.array(posicoes_sensores_otimas)

# Encontra a distribuição ótima dos sensores
posicoes_sensores_otimas = encontrar_distribuicao_otima(num_sensores, area_largura, area_altura, fonte_x, fonte_y, modelo_dispersao)

# Plota os resultados
plt.figure(figsize=(8, 6))
plt.plot(fonte_x, fonte_y, 'ro', label='Fonte de Poluição')
plt.scatter(posicoes_sensores_otimas[:, 0], posicoes_sensores_otimas[:, 1], s=50, c='b', marker='x', label='Sensores Otimizados')
plt.xlabel('Longitude (km)')
plt.ylabel('Latitude (km)')
plt.title('Distribuição Ótima de Sensores para Monitoramento de Poluição')
plt.legend()
plt.grid(True)
plt.show()