import networkx as nx
import numpy as np


def generar_kruskal(grafo: nx.Graph, grid: np.ndarray, pesos_probabilidad: np.ndarray):
    arbol_kruskal = nx.minimum_spanning_tree(grafo, weight='weight')

    for u, v in arbol_kruskal.edges():
        x1, y1 = u
        x2, y2 = v

        # Convertir nodos y pared intermedia a camino
        grid[y1, x1] = 0
        grid[y2, x2] = 0
        pared_x = (x1 + x2) // 2
        pared_y = (y1 + y2) // 2
        grid[pared_y, pared_x] = 0

        # Asignar pesos de probabilidad
        peso = grafo[u][v]['weight']
        pesos_probabilidad[y1, x1] = peso
        pesos_probabilidad[y2, x2] = peso
        pesos_probabilidad[pared_y, pared_x] = peso

    return arbol_kruskal, grid, pesos_probabilidad
