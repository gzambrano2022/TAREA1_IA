from typing import Tuple, List, Optional
from heapq import heappush, heappop


class A_UCS:
    def __init__(self, laberinto, inicio: Tuple[int, int]):
        self.lab = laberinto
        self.pos = inicio
        self.salidas = list(self.lab.salidas)
        self.ruta = []
        self.pasos = 0

    def _ucs(self, objetivo: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Busca camino con UCS"""
        grid = self.lab.grid
        cola = [(0, self.pos, [])]
        visitados = set()

        while cola:
            costo, pos, camino = heappop(cola)
            if pos in visitados:
                continue
            visitados.add(pos)

            if pos == objetivo:
                return camino

            x, y = pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.lab.tamaño and 0 <= ny < self.lab.tamaño and
                        grid[ny, nx] == 0 and (nx, ny) not in visitados):
                    heappush(cola, (costo + 1, (nx, ny), camino + [(nx, ny)]))
        return []

    def mover(self) -> Optional[Tuple[int, int]]:
        """Ejecuta un movimiento"""
        # Re-planificar si no hay ruta o está bloqueada
        if not self.ruta or (self.ruta and self.lab.grid[self.ruta[0][1], self.ruta[0][0]] == 1):
            # Buscar salida más cercana
            objetivo = min(self.salidas, key=lambda s: abs(s[0] - self.pos[0]) + abs(s[1] - self.pos[1]))
            self.ruta = self._ucs(objetivo)

        # Mover
        if self.ruta:
            self.pos = self.ruta.pop(0)
            self.pasos += 1
            return self.pos
        return None

    def meta_alcanzada(self) -> bool:
        if self.pos == self.lab.salida_valida:
            return True
        elif self.pos in self.salidas:
            print(f"Salida falsa en {self.pos}, buscando otra...")
            self.salidas.remove(self.pos)
            self.ruta = []
            return False  # IMPORTANTE: Explícitamente retornar False
        return False  # Para cualquier otro caso