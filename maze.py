import numpy as np
import random
import networkx as nx
import time
import os
from threading import Thread, Lock
from kruskal import generar_kruskal


class Laberinto:
    def __init__(self, tamaño, x_i,y_i, intervalo, num_salidas):
        nodo_inicio=(x_i,y_i)

        # Configuración básica
        self.tamaño = tamaño if tamaño % 2 == 1 else tamaño + 1
        self.nodo_inicio = nodo_inicio if nodo_inicio else (1, 1)
        self.intervalo = intervalo
        self.num_salidas = num_salidas

        # Sistema de salidas
        self.salidas = []
        self.salida_valida = None
        self.salidas_falsas = []
        self.agente_gano = False

        # Control temporal
        self.ultima_actualizacion = time.time()
        self.ejecutando = False
        self.hilo_temporal = None
        self.lock_grid = Lock()

        # Estructuras del laberinto
        self.grafo = nx.Graph()
        self.grid = np.ones((self.tamaño, self.tamaño), dtype=int)
        self.pesos_probabilidad = np.zeros((self.tamaño, self.tamaño))
        self.generacion_completada = False

        # Posición del agente para visualización
        self.posicion_agente = None

        # Inicializar
        self.inicializar_grafo()
        self.generar_salidas()

    def inicializar_grafo(self):
        """Inicializa el grafo con nodos en posiciones impares"""
        # Añadir nodos
        for y in range(1, self.tamaño - 1, 2):
            for x in range(1, self.tamaño - 1, 2):
                self.grafo.add_node((x, y))

        # Añadir aristas con pesos
        for nodo in list(self.grafo.nodes()):
            x, y = nodo
            direcciones = [(2, 0), (0, 2)]

            for dx, dy in direcciones:
                vecino = (x + dx, y + dy)
                if vecino in self.grafo.nodes():
                    peso = self.calcular_peso_probabilidad(x, y, vecino[0], vecino[1])
                    self.grafo.add_edge(nodo, vecino, weight=peso)

        # Inicializar posición de inicio
        x, y = self.nodo_inicio
        self.grid[y, x] = 0
        self.pesos_probabilidad[y, x] = 0.1

    def calcular_peso_probabilidad(self, x1, y1, x2, y2):
        """Calcula pesos basados en distancia al centro"""
        centro = self.tamaño // 2
        dist1 = abs(x1 - centro) + abs(y1 - centro)
        dist2 = abs(x2 - centro) + abs(y2 - centro)
        dist_promedio = (dist1 + dist2) / 2
        return 0.1 + (dist_promedio / self.tamaño) * 0.8

    def generar_salidas(self):
        """Genera k salidas en los bordes, asigna una como válida"""
        posibles_salidas = []

        # Bordes (excluyendo esquinas)
        for x in range(1, self.tamaño - 1, 2):
            posibles_salidas.append((x, 0))  # Superior
            posibles_salidas.append((x, self.tamaño - 1))  # Inferior

        for y in range(1, self.tamaño - 1, 2):
            posibles_salidas.append((0, y))  # Izquierdo
            posibles_salidas.append((self.tamaño - 1, y))  # Derecho

        # Seleccionar salidas aleatorias
        self.salidas = random.sample(posibles_salidas, min(self.num_salidas, len(posibles_salidas)))

        # Asignar salida válida aleatoriamente
        self.salida_valida = random.choice(self.salidas)
        self.salidas_falsas = [s for s in self.salidas if s != self.salida_valida]

        # Crear aberturas para las salidas
        for x, y in self.salidas:
            self.grid[y, x] = 0
            self.pesos_probabilidad[y, x] = 0.05  # Peso bajo para estabilidad

            # Crear conexión hacia el interior
            if y == 0:  # Borde superior
                self.grid[y + 1, x] = 0
                self.pesos_probabilidad[y + 1, x] = 0.1
            elif y == self.tamaño - 1:  # Borde inferior
                self.grid[y - 1, x] = 0
                self.pesos_probabilidad[y - 1, x] = 0.1
            elif x == 0:  # Borde izquierdo
                self.grid[y, x + 1] = 0
                self.pesos_probabilidad[y, x + 1] = 0.1
            elif x == self.tamaño - 1:  # Borde derecho
                self.grid[y, x - 1] = 0
                self.pesos_probabilidad[y, x - 1] = 0.1

    def generar_completamente(self):
        """Genera el laberinto usando Kruskal"""
        with self.lock_grid:
            self.arbol_kruskal, self.grid, self.pesos_probabilidad = generar_kruskal(
                self.grafo, self.grid, self.pesos_probabilidad
            )
            self.mantener_salidas_abiertas()

        self.generacion_completada = True
        self.ultima_actualizacion = time.time()

    def mantener_salidas_abiertas(self):
        """Asegura que las salidas permanezcan abiertas"""
        for x, y in self.salidas:
            self.grid[y, x] = 0
            # Conexión al interior
            if y == 0:
                self.grid[y + 1, x] = 0
            elif y == self.tamaño - 1:
                self.grid[y - 1, x] = 0
            elif x == 0:
                self.grid[y, x + 1] = 0
            elif x == self.tamaño - 1:
                self.grid[y, x - 1] = 0

    def establecer_posicion_agente(self, x_a,y_a):
        """Establece la posición del agente"""
        self.posicion_agente = (x_a,y_a)

    def obtener_intervalo_actual(self):
        """Calcula intervalo basado en inestabilidad"""
        if not self.generacion_completada:
            return self.intervalo

        with self.lock_grid:
            celdas_internas = self.pesos_probabilidad[1:-1, 1:-1]

        mascara_pesos_validos = celdas_internas > 0.05
        if not np.any(mascara_pesos_validos):
            return self.intervalo

        pesos_validos = celdas_internas[mascara_pesos_validos]
        promedio_pesos = np.mean(pesos_validos)

        # Mayor peso = menor intervalo
        factor_ajuste = 1.0 / (1.0 + promedio_pesos * 2)
        intervalo = self.intervalo * factor_ajuste
        return max(0.3, min(5.0, intervalo))

    def actualizacion_temporal(self):
        """Hilo de actualización temporal"""
        while self.ejecutando:
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - self.ultima_actualizacion
            intervalo_requerido = self.obtener_intervalo_actual()

            if tiempo_transcurrido >= intervalo_requerido:
                cambios = self.ejecutar_cambios_dinamicos()
                self.ultima_actualizacion = tiempo_actual

            time.sleep(0.1)

    def ejecutar_cambios_dinamicos(self):
        """Ejecuta los cambios dinámicos del laberinto"""
        cambios = 0

        with self.lock_grid:
            for y in range(1, self.tamaño - 1):
                for x in range(1, self.tamaño - 1):
                    probabilidad = self.pesos_probabilidad[y, x]

                    if random.random() < probabilidad:
                        self.grid[y, x] = 1 - self.grid[y, x]
                        cambios += 1

            # Mantener salidas abiertas
            self.mantener_salidas_abiertas()

        return cambios

    def iniciar_actualizacion_temporal(self):
        """Inicia el sistema temporal"""
        if not self.generacion_completada:
            print("Error: Generar laberinto primero")
            return

        if self.ejecutando:
            print("Actualización ya en ejecución")
            return

        self.ejecutando = True
        self.hilo_temporal = Thread(target=self.actualizacion_temporal)
        self.hilo_temporal.daemon = True
        self.hilo_temporal.start()
        print(f"Actualización temporal iniciada (intervalo base: {self.intervalo}s)")

    def detener_actualizacion_temporal(self):
        """Detiene la actualización temporal"""
        if not self.ejecutando:
            return

        self.ejecutando = False
        if self.hilo_temporal:
            self.hilo_temporal.join(timeout=1.0)
        print("Actualización temporal detenida")

    def imprimir_laberinto(self, usar_colores: bool = True):
        """Imprime el laberinto en consola"""
        # Códigos de color ANSI
        RESET = '\033[0m' if usar_colores else ''
        RED = '\033[91m' if usar_colores else ''
        GREEN = '\033[92m' if usar_colores else ''
        YELLOW = '\033[93m' if usar_colores else ''
        MAGENTA = '\033[95m' if usar_colores else ''

        with self.lock_grid:
            for y in range(self.tamaño):
                fila = ""
                for x in range(self.tamaño):
                    char = ""
                    color = ""

                    if self.posicion_agente and (x, y) == self.posicion_agente:
                        char = "A "
                        color = GREEN
                        # Verificar si está en salida
                        if (x, y) == self.salida_valida:
                            color = GREEN + '\033[5m'  # Parpadeo para victoria
                        elif (x, y) in self.salidas_falsas:
                            color = RED + '\033[5m'  # Parpadeo rojo para trampa
                    elif (x, y) == self.salida_valida:
                        char = "V " if not usar_colores else "✓ "
                        color = YELLOW
                    elif (x, y) in self.salidas_falsas:
                        char = "X " if not usar_colores else "✗ "
                        color = MAGENTA
                    elif self.grid[y, x] == 1:
                        char = "█ "
                    else:
                        char = "  "

                    fila += color + char + RESET

                print(fila)