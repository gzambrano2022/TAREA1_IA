from typing import Tuple, List, Optional
from heapq import heappush, heappop

# Clase que implementa agente que utiliza busqueda de costo uniforme con un mapa dinamico
class A_UCS:

    # Constructor del agente
    def __init__(self, laberinto, inicio: Tuple[int, int]):
        # Instancia del laberinto
        self.lab = laberinto
        # Coordenadas de inicio del agente
        self.pos = inicio
        # Lista de posibles salidas
        self.salidas = list(self.lab.salidas)
        # Ruta hacia el objetivo
        self.ruta = []
        # contador de pasos realizados
        self.pasos = 0

    # Funcion que encuentra el camino mas corto desde la posicion actual hasta el la posicion objetivo
    def _ucs(self, objetivo: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Busca camino con UCS"""
        grid = self.lab.grid
        cola = [(0, self.pos, [])]
        visitados = set()

        #
        while cola:
            costo, pos, camino = heappop(cola)
            if pos in visitados:
                continue
            visitados.add(pos)

            # Si se llega al obj se retorna el camino
            if pos == objetivo:
                return camino

            x, y = pos

            # Explora los vecinos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.lab.tamaño and 0 <= ny < self.lab.tamaño and
                        grid[ny, nx] == 0 and (nx, ny) not in visitados):
                    heappush(cola, (costo + 1, (nx, ny), camino + [(nx, ny)]))
        return []

    # Funcion que realiza un movimiento del agente
    def mover(self) -> Optional[Tuple[int, int]]:
        # Replanifica si no hay ruta o está bloqueada
        if not self.ruta or (self.ruta and self.lab.grid[self.ruta[0][1], self.ruta[0][0]] == 1):
            # Buscar salida ms cercana
            objetivo = min(self.salidas, key=lambda s: abs(s[0] - self.pos[0]) + abs(s[1] - self.pos[1]))
            self.ruta = self._ucs(objetivo)

        # Realiza el siguiente movimiento
        if self.ruta:
            self.pos = self.ruta.pop(0)
            self.pasos += 1
            return self.pos
        return None

    # Verifica si el agente llego al objetivo (Salida real)
    def meta_alcanzada(self) -> bool:

        # Si llega es True
        if self.pos == self.lab.salida_valida:
            return True

        # Si no, o es falsa False
        elif self.pos in self.salidas:
            print(f"Salida falsa en {self.pos}, buscando otra...")
            self.salidas.remove(self.pos)
            self.ruta = []
            return False
        return False