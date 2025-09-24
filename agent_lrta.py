from typing import Tuple, List
import numpy as np

# Clase de Agente Learn Real-Time A*
class AgenteLRTA:
    def __init__(self, laberinto, inicio: Tuple[int, int], objetivo: Tuple[int, int]):
        self.lab = laberinto
        self.pos = inicio
        self.meta = objetivo
        self.H = np.zeros((self.lab.tamaño, self.lab.tamaño))

    def mover(self):
        """Realiza un paso del LRTA*"""
        vecinos = self.lab.obtener_vecinos_validos(self.pos)
        if not vecinos:
            return 0  # No hay movimiento posible

        # Calcular costos locales
        costos = [1 + self.H[v[1], v[0]] for v in vecinos]
        indice_min = np.argmin(costos)
        siguiente = vecinos[indice_min]

        # Actualizar heuristica del nodo actual
        self.H[self.pos[1], self.pos[0]] = costos[indice_min]

        # Mover agente
        self.pos = siguiente
        return self.pos

    def meta_alcanzada(self):
        return self.pos == self.meta
