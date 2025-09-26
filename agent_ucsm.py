from heapq import heappush, heappop

# Clase que implementa agente que utiliza busqueda de costo uniforme con un mapa dinamico
class A_UCS:

    # Constructor del agente
    def __init__(self, laberinto, xi, yi,):
        # Instancia del laberinto
        self.lab = laberinto
        # Coordenadas de inicio del agente
        self.pos = (xi, yi)
        # Lista de posibles salidas
        self.salidas = list(self.lab.salidas)
        # Ruta hacia el objetivo
        self.ruta = []
        # contador de pasos realizados
        self.pasos = 0

    # Funcion que encuentra el camino mas corto desde la posicion actual hasta el la posicion objetivo
    def ucs(self, xo,yo):
        grid = self.lab.grid
        objetivo = (xo,yo)
        cola = [(0, self.pos, [])]
        visitados = set()

        while cola:
            costo, pos, camino = heappop(cola)
            if pos in visitados:
                continue
            visitados.add(pos)

            # Si se llega al obj se retorna el camino
            if pos==objetivo:
                return camino

            x, y = pos

            # Explora los vecinos y añade si la
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.lab.tamaño and 0 <= ny < self.lab.tamaño and grid[ny, nx] == 0 and (nx, ny) not in visitados):
                    heappush(cola, (costo + 1, (nx, ny), camino + [(nx, ny)]))
        return []

    # Funcion que realiza un movimiento del agente
    def mover(self):

        # Replanificar si no hay ruta o si alguna celda de la ruta está bloqueada
        if not self.ruta or any(self.lab.grid[y, x] == 1 for x, y in self.ruta):

            # Intentar todas las salidas disponibles
            rutas_posibles = [(s, self.ucs(s[0],s[1])) for s in self.salidas]
            rutas_posibles = [(s, r) for s, r in rutas_posibles if r]  # solo rutas válidas

            if rutas_posibles:
                objetivo, self.ruta = min(rutas_posibles, key=lambda t: len(t[1]))

            else:
                # Todas las rutas bloqueadas, el agente se queda en su posición
                self.ruta = []

        # Mover al siguiente paso
        if self.ruta:
            self.pos = self.ruta.pop(0)
            self.pasos += 1

        return self.pos

    # Verifica si el agente llego al objetivo (Salida real)
    def meta_alcanzada(self):

        # Si llega es True
        if self.pos == self.lab.salida_valida:
            return True

        # Si no, o es falsa False
        elif self.pos in self.salidas:
            print(f"Salida falsa, busca otra")
            self.salidas.remove(self.pos)
            self.ruta = []
            return False
        return False