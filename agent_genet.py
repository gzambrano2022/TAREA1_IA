import random

# Clase que implementa un algoritmo genetico para la busqueda de la salida de un laberinto
class A_GENET:

    # Constructor de la clase
    def __init__(self,laberinto, xi, yi):
        # Instancia del laberinto
        self.lab = laberinto
        # Coordenadas de inicio del agente
        self.pos = (xi, yi)
        # Arriba, abajo, izquierda y derecha
        self.movimientos=[(0,-1), (0,1),(-1,0), (1,0)]
        #Tamaño de la poblacion
        self.pobla= 40
        # Cantidad de cromosomas
        self.cromosoma = 50
        # Cantidad de generaciones
        self.generacion =100

    # Lista de movimientos aleatorios
    def crear_cromosoma(self):
        cromosoma=[]
        for i in range (self.cromosoma):
            cromosoma.append(random.choice(self.movimientos))
        return cromosoma

    # Conjunto de listas de movimientos
    def creaer_poblacion(self):
        poblacion=[]
        for i in range (self.pobla):
            poblacion.append(self.crear_cromosoma())
        return poblacion

    # Funcion que evalua los cromosomas
    def fitness(self, cromosoma):
        x,y=self.pos
        errores=0

        for movimiento in cromosoma:
            dx,dy=movimiento
            nx,ny=x + dx, y + dy

            # Verifica si se excede del tablero
            if nx < 0 or nx >= self.lab.tamaño or ny < 0 or ny >= self.lab.tamaño:
                errores += 1
                continue

            # Verificar si choca con una pared
            if self.lab.grid[ny, nx] == 1:
                errores += 1
            else:
                x, y = nx, ny

            # Verificar si llego a alguna salida, si es falsa penalizacion de +5 si es verdadera termina
            if (x, y) in self.lab.salidas_falsas:
                errores += 5
            elif (x, y) == self.lab.salida_valida:
                break

        return errores

    # Seleccion por torneo
    def seleccionar(self, poblacion):

        participantes = random.sample(poblacion, 3)
        ganador = min(participantes, key=lambda ind: self.fitness(ind))

        return ganador

    # Realiza el cruce mediante el metodo one-point
    def cruzar(self, padre1, padre2):

        # Punto de cruce aleatorio
        punto= random.randint(1,self.cromosoma-1)

        # Hijos
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]

        return hijo1, hijo2

    # Cambios aleatorios
    def mutar(self, cromosoma):

        # Indice a cambear
        indice=random.randint(0,len(cromosoma)-1)

        # Cambio de valor
        cromosoma[indice]= random.choice(self.movimientos)

        return cromosoma

    # Proceso genetico
    def evolucionar(self):
        # Crear población inicial
        poblacion = self.creaer_poblacion()

        for gen in range(self.generacion):
            nueva_poblacion = []

            # Generar nueva población
            while len(nueva_poblacion) < self.pobla:
                # Selección por torneo
                padre1 = self.seleccionar(poblacion)
                padre2 = self.seleccionar(poblacion)

                # Cruce
                hijo1, hijo2 = self.cruzar(padre1, padre2)

                # Mutación con probabilidad (por ejemplo 20%)
                if random.random() < 0.2:
                    hijo1 = self.mutar(hijo1)
                if random.random() < 0.2:
                    hijo2 = self.mutar(hijo2)

                # Agregar hijos a la nueva población
                nueva_poblacion.extend([hijo1, hijo2])

            # Limitar población a tamaño original
            poblacion = nueva_poblacion[:self.pobla]

            # Opcional: mostrar mejor fitness de la generación
            mejor = min(poblacion, key=lambda c: self.fitness(c))
            print(f"Generación {gen + 1}: Mejor fitness = {self.fitness(mejor)}")

        # Retornar mejor cromosoma encontrado
        mejor_final = min(poblacion, key=lambda c: self.fitness(c))
        return mejor_final

