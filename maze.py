import numpy as np
import random
import networkx as nx
import time
import os
from typing import List, Tuple, Dict, Callable, Optional
from threading import Thread, Lock
from datetime import datetime


class LaberintoDinamicoTemporal:
    def __init__(self, tamaño: int, nodo_inicio: Tuple[int, int] = None,
                 intervalo_base: float = 2.0, auto_visualizar: bool = True,
                 num_salidas: int = 3):
        """
        Laberinto dinámico con actualización por tiempo basada en pesos

        Args:
            tamaño: Tamaño de la grid (debe ser impar)
            nodo_inicio: Tupla (x, y) donde empezará el agente
            intervalo_base: Intervalo base en segundos para actualizaciones
            auto_visualizar: Si True, mostrará automáticamente los cambios
            num_salidas: Número total de salidas (k salidas, solo 1 válida)
        """
        # Configuración básica
        self.tamaño = tamaño if tamaño % 2 == 1 else tamaño + 1
        self.nodo_inicio = nodo_inicio if nodo_inicio else (1, 1)
        self.intervalo_base = intervalo_base
        self.auto_visualizar = auto_visualizar
        self.num_salidas = max(1, num_salidas)  # Al menos 1 salida

        # Sistema de salidas
        self.salidas = []  # Lista de posiciones (x, y) de salidas
        self.salida_valida = None  # La única salida correcta
        self.salidas_falsas = []  # Salidas trampa
        self.agente_gano = False  # Si el agente llegó a la salida válida

        # Sistema de reparación de caminos
        self.reparacion_activa = True  # Si está activada la reparación automática
        self.caminos_criticos = set()  # Caminos que no deben bloquearse
        self.historial_reparaciones = []  # Log de reparaciones realizadas

        # Control de tiempo
        self.ultima_actualizacion = time.time()
        self.tiempo_acumulado = 0.0
        self.ejecutando = False
        self.hilo_temporal = None
        self.lock_grid = Lock()  # Para thread safety

        # Estructuras del laberinto
        self.grafo = nx.Graph()
        self.grid = np.ones((self.tamaño, self.tamaño), dtype=int)
        self.grid_anterior = None  # Para detectar cambios
        self.pesos_probabilidad = np.zeros((self.tamaño, self.tamaño))
        self.generacion_completada = False

        # Estadísticas y visualización
        self.cambios_totales = 0
        self.historial_cambios = []
        self.posicion_agente = None
        self.callback_actualizacion = None
        self.mostrar_cambios_destacados = True
        self.ultimos_cambios = set()  # Para highlighting

        self._inicializar_grafo()
        self._generar_salidas()

    def _inicializar_grafo(self):
        """Inicializa el grafo con NetworkX"""
        # Añadir nodos en posiciones impares
        for y in range(1, self.tamaño - 1, 2):
            for x in range(1, self.tamaño - 1, 2):
                self.grafo.add_node((x, y))

        # Añadir aristas con pesos probabilísticos
        for nodo in list(self.grafo.nodes()):
            x, y = nodo
            direcciones = [(2, 0), (0, 2)]

            for dx, dy in direcciones:
                vecino = (x + dx, y + dy)
                if vecino in self.grafo.nodes():
                    peso = self._calcular_peso_probabilidad(x, y, vecino[0], vecino[1])
                    self.grafo.add_edge(nodo, vecino, weight=peso)

        # Inicializar nodo inicio
        x, y = self.nodo_inicio
        self.grid[y, x] = 0
        self.pesos_probabilidad[y, x] = 0.1

    def _generar_salidas(self):
        """Genera k salidas en los bordes del laberinto, asigna una como válida"""
        # Posibles posiciones de salida en los bordes (excluyendo esquinas)
        posibles_salidas = []

        # Borde superior e inferior
        for x in range(1, self.tamaño - 1, 2):
            posibles_salidas.append((x, 0))  # Borde superior
            posibles_salidas.append((x, self.tamaño - 1))  # Borde inferior

        # Borde izquierdo y derecho
        for y in range(1, self.tamaño - 1, 2):
            posibles_salidas.append((0, y))  # Borde izquierdo
            posibles_salidas.append((self.tamaño - 1, y))  # Borde derecho

        # Seleccionar k salidas aleatorias
        self.salidas = random.sample(posibles_salidas, min(self.num_salidas, len(posibles_salidas)))

        # Asignar aleatoriamente la salida válida
        self.salida_valida = random.choice(self.salidas)
        self.salidas_falsas = [s for s in self.salidas if s != self.salida_valida]

        # Crear aberturas en el borde para las salidas
        for x, y in self.salidas:
            self.grid[y, x] = 0  # Hacer la salida transitable
            self.pesos_probabilidad[y, x] = 0.05  # Peso muy bajo para estabilidad

            # Crear conexión hacia el interior del laberinto
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

        # Marcar caminos críticos iniciales
        self._identificar_caminos_criticos()

    def _calcular_peso_probabilidad(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calcula pesos basados en distancia al centro"""
        centro = self.tamaño // 2
        dist1 = abs(x1 - centro) + abs(y1 - centro)
        dist2 = abs(x2 - centro) + abs(y2 - centro)
        dist_promedio = (dist1 + dist2) / 2
        return 0.1 + (dist_promedio / self.tamaño) * 0.8

    def establecer_posicion_agente(self, posicion: Tuple[int, int]):
        """Establece la posición del agente para visualización"""
        posicion_anterior = self.posicion_agente
        self.posicion_agente = posicion
        self._verificar_victoria()

        # Verificar conectividad después del movimiento
        if self.reparacion_activa and posicion_anterior != posicion:
            self._verificar_y_reparar_conectividad()

    def _verificar_y_reparar_conectividad(self):
        """Verifica si el agente tiene conexión a las salidas y repara si es necesario"""
        if not self.posicion_agente or not self.generacion_completada:
            return

        # Verificar conectividad a todas las salidas
        salidas_accesibles = self._obtener_salidas_accesibles()

        if len(salidas_accesibles) == 0:
            # ¡El agente está completamente bloqueado!
            print("🚨 AGENTE BLOQUEADO - Iniciando reparación de emergencia...")
            self._reparacion_emergencia()
        elif self.salida_valida not in salidas_accesibles:
            # La salida válida no es accesible
            print("⚠️  SALIDA VÁLIDA INACCESIBLE - Creando camino alternativo...")
            self._crear_camino_a_salida_valida()

    def _obtener_salidas_accesibles(self) -> List[Tuple[int, int]]:
        """Retorna lista de salidas accesibles desde la posición del agente usando BFS"""
        if not self.posicion_agente:
            return []

        visitados = set()
        cola = [self.posicion_agente]
        salidas_encontradas = []

        while cola:
            actual = cola.pop(0)
            if actual in visitados:
                continue

            visitados.add(actual)

            # Si es una salida, añadirla
            if actual in self.salidas:
                salidas_encontradas.append(actual)

            # Explorar vecinos
            x, y = actual
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.tamaño and 0 <= ny < self.tamaño and
                        (nx, ny) not in visitados and self.grid[ny, nx] == 0):
                    cola.append((nx, ny))

        return salidas_encontradas

    def _reparacion_emergencia(self):
        """Reparación de emergencia: crea un camino directo a la salida más cercana"""
        salida_objetivo = self._obtener_salida_mas_cercana()
        if not salida_objetivo:
            return

        # Crear camino directo usando línea de Bresenham modificada
        camino_reparacion = self._calcular_camino_directo(self.posicion_agente, salida_objetivo)

        reparaciones = 0
        for x, y in camino_reparacion:
            if 0 <= x < self.tamaño and 0 <= y < self.tamaño:
                if self.grid[y, x] == 1:  # Si es pared
                    self.grid[y, x] = 0  # Convertir a camino
                    self.pesos_probabilidad[y, x] = 0.05  # Peso muy bajo
                    self.caminos_criticos.add((x, y))  # Marcar como crítico
                    reparaciones += 1

        self.historial_reparaciones.append({
            'timestamp': time.time(),
            'tipo': 'emergencia',
            'desde': self.posicion_agente,
            'hacia': salida_objetivo,
            'reparaciones': reparaciones,
            'camino': camino_reparacion
        })

        print(f"✅ Reparación completada: {reparaciones} paredes removidas")

    def _crear_camino_a_salida_valida(self):
        """Crea un camino específico hacia la salida válida si no es accesible"""
        if not self.salida_valida:
            return

        # Intentar encontrar un camino más inteligente
        camino_optimo = self._buscar_camino_optimo(self.posicion_agente, self.salida_valida)

        if not camino_optimo:
            # Si no encuentra camino óptimo, usar línea directa
            camino_optimo = self._calcular_camino_directo(self.posicion_agente, self.salida_valida)

        reparaciones = 0
        for x, y in camino_optimo:
            if (0 <= x < self.tamaño and 0 <= y < self.tamaño and
                    self.grid[y, x] == 1):
                self.grid[y, x] = 0
                self.pesos_probabilidad[y, x] = 0.08  # Peso bajo pero no crítico
                reparaciones += 1

        self.historial_reparaciones.append({
            'timestamp': time.time(),
            'tipo': 'salida_valida',
            'desde': self.posicion_agente,
            'hacia': self.salida_valida,
            'reparaciones': reparaciones,
            'camino': camino_optimo
        })

        print(f"🎯 Camino a salida válida creado: {reparaciones} conexiones")

    def _obtener_salida_mas_cercana(self) -> Tuple[int, int]:
        """Retorna la salida más cercana al agente"""
        if not self.posicion_agente or not self.salidas:
            return None

        distancias = []
        for salida in self.salidas:
            dist = self._calcular_distancia_manhattan(self.posicion_agente, salida)
            distancias.append((dist, salida))

        return min(distancias)[1]

    def _calcular_camino_directo(self, inicio: Tuple[int, int], fin: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Calcula un camino directo entre dos puntos usando algoritmo de línea de Bresenham"""
        x0, y0 = inicio
        x1, y1 = fin

        camino = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1

        error = dx - dy
        x, y = x0, y0

        while True:
            camino.append((x, y))

            if x == x1 and y == y1:
                break

            error2 = 2 * error

            if error2 > -dy:
                error -= dy
                x += x_step

            if error2 < dx:
                error += dx
                y += y_step

        return camino

    def _buscar_camino_optimo(self, inicio: Tuple[int, int], fin: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Busca un camino óptimo evitando crear demasiadas aberturas
        Usa A* simplificado considerando paredes existentes
        """
        # Implementación simplificada de A*
        from heapq import heappush, heappop

        def heuristica(pos):
            return self._calcular_distancia_manhattan(pos, fin)

        cola_prioridad = [(0, inicio, [])]
        visitados = set()

        while cola_prioridad:
            f_cost, actual, camino = heappop(cola_prioridad)

            if actual in visitados:
                continue

            visitados.add(actual)
            nuevo_camino = camino + [actual]

            if actual == fin:
                return nuevo_camino

            # Limitar búsqueda para evitar computación excesiva
            if len(nuevo_camino) > 50:
                continue

            x, y = actual
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if (0 <= nx < self.tamaño and 0 <= ny < self.tamaño and
                        (nx, ny) not in visitados):
                    # Costo: 1 si es camino libre, 10 si requiere romper pared
                    costo = 1 if self.grid[ny, nx] == 0 else 10
                    g_cost = len(nuevo_camino) + costo
                    h_cost = heuristica((nx, ny))
                    f_cost = g_cost + h_cost

                    heappush(cola_prioridad, (f_cost, (nx, ny), nuevo_camino))

        return []  # No se encontró camino

    def _identificar_caminos_criticos(self):
        """Identifica caminos críticos que nunca deben bloquearse"""
        # Agregar posición inicial y salidas como críticos
        self.caminos_criticos.add(self.nodo_inicio)
        for salida in self.salidas:
            self.caminos_criticos.add(salida)

        # Agregar conexiones inmediatas a salidas
        for x, y in self.salidas:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.tamaño and 0 <= ny < self.tamaño and
                        self.grid[ny, nx] == 0):
                    self.caminos_criticos.add((nx, ny))

    def configurar_reparacion(self, activa: bool = True):
        """Configura el sistema de reparación automática"""
        self.reparacion_activa = activa
        print(f"🔧 Sistema de reparación: {'ACTIVADO' if activa else 'DESACTIVADO'}")

    def obtener_estadisticas_reparacion(self) -> Dict:
        """Retorna estadísticas del sistema de reparación"""
        if not self.historial_reparaciones:
            return {
                'reparaciones_totales': 0,
                'reparaciones_emergencia': 0,
                'reparaciones_salida_valida': 0,
                'caminos_criticos': len(self.caminos_criticos),
                'sistema_activo': self.reparacion_activa
            }

        tipos = [r['tipo'] for r in self.historial_reparaciones]
        return {
            'reparaciones_totales': len(self.historial_reparaciones),
            'reparaciones_emergencia': tipos.count('emergencia'),
            'reparaciones_salida_valida': tipos.count('salida_valida'),
            'caminos_criticos': len(self.caminos_criticos),
            'sistema_activo': self.reparacion_activa,
            'ultima_reparacion': self.historial_reparaciones[-1] if self.historial_reparaciones else None
        }

    def _verificar_victoria(self):
        """Verifica si el agente ha llegado a la salida válida"""
        if self.posicion_agente and self.posicion_agente == self.salida_valida:
            if not self.agente_gano:
                self.agente_gano = True
                print(f"\n🎉 ¡VICTORIA! El agente llegó a la salida válida en {self.salida_valida}")
                if self.ejecutando:
                    self.detener_actualizacion_temporal()
        elif self.posicion_agente and self.posicion_agente in self.salidas_falsas:
            print(f"\n💀 ¡SALIDA FALSA! El agente cayó en una trampa en {self.posicion_agente}")

    def obtener_info_salidas(self) -> Dict:
        """Retorna información sobre las salidas del laberinto"""
        return {
            'total_salidas': len(self.salidas),
            'posiciones_salidas': self.salidas,
            'salida_valida': self.salida_valida,
            'salidas_falsas': self.salidas_falsas,
            'agente_gano': self.agente_gano,
            'distancia_a_salida_valida': self._calcular_distancia_manhattan(
                self.posicion_agente, self.salida_valida) if self.posicion_agente else None
        }

    def _calcular_distancia_manhattan(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcula distancia Manhattan entre dos posiciones"""
        if not pos1 or not pos2:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reiniciar_juego(self):
        """Reinicia el estado del juego manteniendo la estructura del laberinto"""
        self.agente_gano = False
        self.establecer_posicion_agente(self.nodo_inicio)
        # Reasignar salida válida aleatoriamente
        self.salida_valida = random.choice(self.salidas)
        self.salidas_falsas = [s for s in self.salidas if s != self.salida_valida]
        print(f"🔄 Juego reiniciado. Nueva salida válida asignada.")

    def establecer_callback_actualizacion(self, callback: Callable):
        """Establece una función callback que se ejecuta después de cada actualización"""
        self.callback_actualizacion = callback

    def generar_completamente(self):
        """Genera el laberinto completo usando algoritmo de Kruskal"""
        with self.lock_grid:
            # Crear árbol de expansión mínima usando pesos probabilísticos
            self.arbol_kruskal = nx.minimum_spanning_tree(self.grafo, weight='weight')

            for u, v in self.arbol_kruskal.edges():
                x1, y1 = u
                x2, y2 = v

                # Convertir nodos y pared intermedia a camino
                self.grid[y1, x1] = 0
                self.grid[y2, x2] = 0

                pared_x = (x1 + x2) // 2
                pared_y = (y1 + y2) // 2
                self.grid[pared_y, pared_x] = 0

                # Asignar pesos de probabilidad
                peso = self.grafo[u][v]['weight']
                self.pesos_probabilidad[y1, x1] = peso
                self.pesos_probabilidad[y2, x2] = peso
                self.pesos_probabilidad[pared_y, pared_x] = peso

            # Guardar copia para detectar cambios
            self.grid_anterior = self.grid.copy()

            # Asegurar que las salidas permanezcan abiertas
            self._mantener_salidas_abiertas()

        self.generacion_completada = True
        self.ultima_actualizacion = time.time()

    def _mantener_salidas_abiertas(self):
        """Asegura que las salidas permanezcan siempre abiertas"""
        for x, y in self.salidas:
            self.grid[y, x] = 0  # Forzar que la salida esté abierta
            # Asegurar conexión al interior
            if y == 0:  # Borde superior
                self.grid[y + 1, x] = 0
            elif y == self.tamaño - 1:  # Borde inferior
                self.grid[y - 1, x] = 0
            elif x == 0:  # Borde izquierdo
                self.grid[y, x + 1] = 0
            elif x == self.tamaño - 1:  # Borde derecho
                self.grid[y, x - 1] = 0

    def _obtener_intervalo_actual(self) -> float:
        """Calcula el intervalo actual basado en la inestabilidad del laberinto"""
        if not self.generacion_completada:
            return self.intervalo_base

        # Solo celdas internas (no bordes)
        with self.lock_grid:
            celdas_internas = self.pesos_probabilidad[1:-1, 1:-1]

        # Filtrar celdas con peso relevante (evitar ceros de bordes)
        mascara_pesos_validos = celdas_internas > 0.05
        if not np.any(mascara_pesos_validos):
            return self.intervalo_base

        pesos_validos = celdas_internas[mascara_pesos_validos]
        promedio_pesos = np.mean(pesos_validos)

        # Mayor peso = menor intervalo (más frecuente)
        factor_ajuste = 1.0 / (1.0 + promedio_pesos * 2)
        intervalo = self.intervalo_base * factor_ajuste
        return max(0.3, min(5.0, intervalo))

    def _limpiar_pantalla(self):
        """Limpia la pantalla de manera multiplataforma"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _detectar_cambios(self) -> set:
        """Detecta las posiciones que cambiaron desde la última actualización"""
        if self.grid_anterior is None:
            return set()

        cambios = set()
        for y in range(self.tamaño):
            for x in range(self.tamaño):
                if self.grid[y, x] != self.grid_anterior[y, x]:
                    cambios.add((x, y))
        return cambios

    def _actualizacion_temporal(self):
        """Hilo principal de actualización por tiempo"""
        while self.ejecutando:
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - self.ultima_actualizacion
            intervalo_requerido = self._obtener_intervalo_actual()

            if tiempo_transcurrido >= intervalo_requerido:
                cambios = self._ejecutar_actualizacion_con_pesos()
                self.ultima_actualizacion = tiempo_actual
                self.tiempo_acumulado += tiempo_transcurrido

                # Detectar cambios para highlighting
                self.ultimos_cambios = self._detectar_cambios()

                # Registrar cambio
                registro = {
                    'timestamp': tiempo_actual,
                    'intervalo_usado': intervalo_requerido,
                    'cambios': cambios,
                    'peso_promedio': np.mean(self.pesos_probabilidad[1:-1, 1:-1]),
                    'posiciones_cambiadas': len(self.ultimos_cambios)
                }
                self.historial_cambios.append(registro)

                # Mostrar automáticamente si está habilitado
                if self.auto_visualizar:
                    self._limpiar_pantalla()
                    self._imprimir_estado_completo(registro, self.ultimos_cambios)

                # Ejecutar callback si existe
                if self.callback_actualizacion:
                    self.callback_actualizacion(self, registro)

            time.sleep(0.1)

    def _ejecutar_actualizacion_con_pesos(self) -> int:
        """Ejecuta actualización considerando los pesos de probabilidad"""
        cambios = 0

        with self.lock_grid:
            # Guardar estado anterior
            self.grid_anterior = self.grid.copy()

            for y in range(1, self.tamaño - 1):
                for x in range(1, self.tamaño - 1):
                    probabilidad = self.pesos_probabilidad[y, x]

                    # Aplicar cambio basado en probabilidad del peso
                    if random.random() < probabilidad:
                        self.grid[y, x] = 1 - self.grid[y, x]
                        cambios += 1

            # Mantener las salidas siempre abiertas después de los cambios
            self._mantener_salidas_abiertas()

            # Proteger caminos críticos
            self._proteger_caminos_criticos()

            # Verificar conectividad del agente si la reparación está activa
            if self.reparacion_activa and self.posicion_agente:
                self._verificar_conectividad_post_cambios()

        self.cambios_totales += cambios
        return cambios

    def _proteger_caminos_criticos(self):
        """Protege los caminos críticos de ser bloqueados"""
        for x, y in self.caminos_criticos:
            if 0 <= x < self.tamaño and 0 <= y < self.tamaño:
                self.grid[y, x] = 0  # Forzar que sea camino

    def _verificar_conectividad_post_cambios(self):
        """Verifica conectividad después de cambios dinámicos y repara si es necesario"""
        salidas_accesibles = self._obtener_salidas_accesibles()

        # Si no hay acceso a ninguna salida, reparación de emergencia
        if len(salidas_accesibles) == 0:
            print("🚨 CONEXIÓN PERDIDA - Reparando automáticamente...")
            self._reparacion_emergencia()
        # Si la salida válida no es accesible, crear camino
        elif self.salida_valida and self.salida_valida not in salidas_accesibles:
            print("⚠️  SALIDA VÁLIDA BLOQUEADA - Reparando...")
            self._crear_camino_a_salida_valida()

    def _imprimir_estado_completo(self, registro: Dict, cambios_destacados: set):
        """Imprime el estado completo del laberinto con información adicional"""
        print("=" * 70)
        print(f"🕐 {datetime.now().strftime('%H:%M:%S')} | Actualización #{len(self.historial_cambios)}")
        print(
            f"⏱️  Intervalo: {registro['intervalo_usado']:.1f}s | Cambios: {registro['cambios']} | Posiciones: {registro['posiciones_cambiadas']}")
        print(f"📊 Peso promedio: {registro['peso_promedio']:.3f} | Total cambios: {self.cambios_totales}")

        # Información de salidas
        info_salidas = self.obtener_info_salidas()
        print(
            f"🎯 Salidas: {info_salidas['total_salidas']} | Válida: {info_salidas['salida_valida']} | Victoria: {'Sí' if info_salidas['agente_gano'] else 'No'}")
        if info_salidas['distancia_a_salida_valida'] is not None:
            print(f"📏 Distancia a salida válida: {info_salidas['distancia_a_salida_valida']}")

        # Información de reparación si está activa
        if self.reparacion_activa:
            stats_reparacion = self.obtener_estadisticas_reparacion()
            if stats_reparacion['reparaciones_totales'] > 0:
                print(
                    f"🔧 Reparaciones: {stats_reparacion['reparaciones_totales']} | Críticos: {stats_reparacion['caminos_criticos']}")

        print("=" * 70)

        self.imprimir_laberinto(
            posicion_agente=self.posicion_agente,
            destacar_cambios=cambios_destacados if self.mostrar_cambios_destacados else None
        )

        if cambios_destacados and self.mostrar_cambios_destacados:
            print(f"\n🔥 Cambios destacados en rojo: {len(cambios_destacados)} posiciones")

    def iniciar_actualizacion_temporal(self, limpiar_pantalla_inicial: bool = True):
        """Inicia el sistema de actualización por tiempo"""
        if not self.generacion_completada:
            print("❌ Error: Generar laberinto primero")
            return

        if self.ejecutando:
            print("⚠️  Actualización ya en ejecución")
            return

        if limpiar_pantalla_inicial and self.auto_visualizar:
            self._limpiar_pantalla()

        self.ejecutando = True
        self.hilo_temporal = Thread(target=self._actualizacion_temporal)
        self.hilo_temporal.daemon = True
        self.hilo_temporal.start()

        print(f"✅ Actualización temporal iniciada (base: {self.intervalo_base}s, auto-visual: {self.auto_visualizar})")

        if self.auto_visualizar:
            print("💡 El laberinto se actualizará automáticamente en pantalla")
            print("   Presiona Ctrl+C para detener\n")

    def detener_actualizacion_temporal(self):
        """Detiene la actualización por tiempo"""
        if not self.ejecutando:
            return

        self.ejecutando = False
        if self.hilo_temporal:
            self.hilo_temporal.join(timeout=1.0)
        print("\n⏹️ Actualización temporal detenida")

    def actualizar_manual_por_tiempo(self, tiempo_transcurrido: float, mostrar_resultado: bool = True) -> int:
        """
        Actualización manual para simulaciones controladas

        Args:
            tiempo_transcurrido: Segundos desde última actualización
            mostrar_resultado: Si mostrar el laberinto después del cambio
        Returns:
            Número de cambios aplicados
        """
        if not self.generacion_completada:
            return 0

        intervalo_requerido = self._obtener_intervalo_actual()

        if tiempo_transcurrido >= intervalo_requerido:
            cambios = self._ejecutar_actualizacion_con_pesos()
            self.ultima_actualizacion = time.time()
            self.tiempo_acumulado += tiempo_transcurrido

            # Detectar cambios
            cambios_posiciones = self._detectar_cambios()
            self.ultimos_cambios = cambios_posiciones

            registro = {
                'timestamp': time.time(),
                'intervalo_usado': intervalo_requerido,
                'cambios': cambios,
                'peso_promedio': np.mean(self.pesos_probabilidad[1:-1, 1:-1]),
                'posiciones_cambiadas': len(cambios_posiciones)
            }

            self.historial_cambios.append(registro)

            if mostrar_resultado:
                print(f"\n⏰ Actualización manual (t={tiempo_transcurrido:.1f}s)")
                self.imprimir_laberinto(
                    posicion_agente=self.posicion_agente,
                    destacar_cambios=cambios_posiciones if self.mostrar_cambios_destacados else None
                )

            return cambios

        return 0

    def obtener_estadisticas_temporales(self) -> Dict:
        """Retorna estadísticas del sistema temporal"""
        if self.historial_cambios:
            ultimo = self.historial_cambios[-1]
            intervalo_promedio = np.mean([c['intervalo_usado'] for c in self.historial_cambios])
        else:
            ultimo = {}
            intervalo_promedio = self.intervalo_base

        return {
            'ejecutando': self.ejecutando,
            'cambios_totales': self.cambios_totales,
            'tiempo_acumulado': self.tiempo_acumulado,
            'intervalo_actual': self._obtener_intervalo_actual(),
            'intervalo_promedio': intervalo_promedio,
            'total_actualizaciones': len(self.historial_cambios),
            'peso_promedio_actual': np.mean(self.pesos_probabilidad[1:-1, 1:-1]),
            'ultima_actualizacion': ultimo
        }

    def obtener_vecinos_validos(self, posicion: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Obtiene vecinos válidos para movimiento"""
        if not hasattr(self, 'arbol_kruskal') or posicion not in self.arbol_kruskal:
            return []

        with self.lock_grid:
            vecinos = list(self.arbol_kruskal.neighbors(posicion))
            return [v for v in vecinos if self.grid[v[1], v[0]] == 0]

    def imprimir_laberinto(self, posicion_agente: Tuple[int, int] = None,
                           destacar_cambios: set = None, usar_colores: bool = True):
        """
        Imprime el laberinto en consola con opciones de destacado

        Args:
            posicion_agente: Posición del agente (x, y)
            destacar_cambios: Set de posiciones (x, y) para destacar
            usar_colores: Si usar códigos de color ANSI
        """
        # Códigos de color ANSI
        RESET = '\033[0m' if usar_colores else ''
        RED = '\033[91m' if usar_colores else ''
        GREEN = '\033[92m' if usar_colores else ''
        BLUE = '\033[94m' if usar_colores else ''
        YELLOW = '\033[93m' if usar_colores else ''
        MAGENTA = '\033[95m' if usar_colores else ''
        CYAN = '\033[96m' if usar_colores else ''

        with self.lock_grid:
            for y in range(self.tamaño):
                fila = ""
                for x in range(self.tamaño):
                    char = ""
                    color = ""

                    # Determinar carácter y color
                    if posicion_agente and (x, y) == posicion_agente:
                        char = "A "
                        color = GREEN
                        # Verificar si está en salida válida o falsa
                        if (x, y) == self.salida_valida:
                            color = GREEN + '\033[5m'  # Parpadeo para victoria
                        elif (x, y) in self.salidas_falsas:
                            color = RED + '\033[5m'  # Parpadeo rojo para trampa
                    elif (x, y) == self.salida_valida:
                        char = "✓ " if usar_colores else "V "
                        color = YELLOW  # Salida válida en amarillo (oculta)
                    elif (x, y) in self.salidas_falsas:
                        char = "✗ " if usar_colores else "X "
                        color = MAGENTA  # Salidas falsas en magenta (oculta)
                    elif (x, y) in self.caminos_criticos:
                        # Destacar caminos críticos
                        if usar_colores:
                            color = CYAN
                            char = "▓ "  # Patrón especial para caminos críticos
                        else:
                            char = "░ "
                    elif self.grid[y, x] == 1:
                        char = "█ "
                        # Destacar cambios en rojo
                        if destacar_cambios and (x, y) in destacar_cambios:
                            color = RED
                    else:
                        char = "  "
                        # Destacar caminos nuevos en azul
                        if destacar_cambios and (x, y) in destacar_cambios:
                            color = BLUE
                            char = "░ "

                    fila += color + char + RESET

                print(fila)

    def modo_interactivo(self):
        """Modo interactivo para controlar el laberinto"""
        if not self.generacion_completada:
            print("❌ Generar laberinto primero")
            return

        print("\n🎮 Modo Interactivo del Laberinto Dinámico")
        print("Comandos:")
        print("  's' - Iniciar/detener actualización automática")
        print("  'u' - Actualizar manualmente")
        print("  'a' - Mover agente (WASD)")
        print("  'i' - Mostrar información de salidas")
        print("  'g' - Reiniciar juego (nueva salida válida)")
        print("  'c' - Alternar destacado de cambios")
        print("  'r' - Alternar sistema de reparación")
        print("  'e' - Estadísticas de reparación")
        print("  'p' - Mostrar estadísticas")
        print("  't' - Reiniciar laberinto completo")
        print("  'q' - Salir")
        print("-" * 50)

        self.establecer_posicion_agente(self.nodo_inicio)
        self.imprimir_laberinto()
        self._mostrar_info_salidas()

        try:
            while True:
                if self.agente_gano:
                    print("\n🎉 ¡Has ganado! Usa 'g' para jugar de nuevo o 'q' para salir.")

                cmd = input("\n> ").lower().strip()

                if cmd == 'q':
                    break
                elif cmd == 's':
                    if self.ejecutando:
                        self.detener_actualizacion_temporal()
                    else:
                        if not self.agente_gano:
                            self.iniciar_actualizacion_temporal()
                        else:
                            print("🎉 Juego terminado. Usa 'g' para reiniciar.")
                elif cmd == 'u':
                    cambios = self.actualizar_manual_por_tiempo(self.intervalo_base)
                    print(f"Cambios aplicados: {cambios}")
                elif cmd == 'a':
                    self._mover_agente_interactivo()
                elif cmd == 'i':
                    self._mostrar_info_salidas()
                elif cmd == 'g':
                    self.detener_actualizacion_temporal()
                    self.reiniciar_juego()
                    self.imprimir_laberinto()
                elif cmd == 'c':
                    self.mostrar_cambios_destacados = not self.mostrar_cambios_destacados
                    print(f"Destacado de cambios: {'ON' if self.mostrar_cambios_destacados else 'OFF'}")
                elif cmd == 'r':
                    self.configurar_reparacion(not self.reparacion_activa)
                elif cmd == 'e':
                    stats_rep = self.obtener_estadisticas_reparacion()
                    print("🔧 Estadísticas de Reparación:")
                    for k, v in stats_rep.items():
                        print(f"   {k}: {v}")
                elif cmd == 'p':
                    stats = self.obtener_estadisticas_temporales()
                    for k, v in stats.items():
                        print(f"  {k}: {v}")
                elif cmd == 't':
                    self.detener_actualizacion_temporal()
                    self.__init__(self.tamaño, self.nodo_inicio, self.intervalo_base,
                                  self.auto_visualizar, self.num_salidas)
                    self.generar_completamente()
                    print("🔄 Laberinto completamente reiniciado")
                    self.imprimir_laberinto()
                    self._mostrar_info_salidas()
                else:
                    print("❓ Comando no reconocido")

        except KeyboardInterrupt:
            self.detener_actualizacion_temporal()
            print("\n👋 Saliendo...")

    def _mostrar_info_salidas(self):
        """Muestra información sobre las salidas del laberinto"""
        info = self.obtener_info_salidas()
        print(f"\n🎯 Información de Salidas:")
        print(f"   Total: {info['total_salidas']} salidas")
        print(f"   Posiciones: {info['posiciones_salidas']}")
        print(f"   ✓ = Salida válida (amarillo)")
        print(f"   ✗ = Salidas falsas (magenta)")
        print(f"   ▓ = Caminos críticos protegidos (cyan)")
        print(f"   🔧 Sistema de reparación: {'ACTIVO' if self.reparacion_activa else 'INACTIVO'}")
        print(f"   Victoria: {'Sí' if info['agente_gano'] else 'No'}")
        if info['distancia_a_salida_valida']:
            print(f"   Distancia a salida válida: {info['distancia_a_salida_valida']}")

        stats_rep = self.obtener_estadisticas_reparacion()
        if stats_rep['reparaciones_totales'] > 0:
            print(f"   Reparaciones realizadas: {stats_rep['reparaciones_totales']}")

    def _mover_agente_interactivo(self):
        """Permite mover el agente interactivamente"""
        if self.agente_gano:
            print("🎉 El juego ya terminó. Usa 'g' para reiniciar.")
            return

        print("Mover agente: W(arriba), A(izq), S(abajo), D(der), X(cancelar)")
        direccion = input("Dirección: ").lower().strip()

        if direccion == 'x':
            return

        movimientos = {
            'w': (0, -1), 'a': (-1, 0), 's': (0, 1), 'd': (1, 0)
        }

        if direccion not in movimientos:
            print("❓ Dirección inválida")
            return

        if not self.posicion_agente:
            print("❌ Posición del agente no establecida")
            return

        dx, dy = movimientos[direccion]
        nueva_x = self.posicion_agente[0] + dx
        nueva_y = self.posicion_agente[1] + dy

        # Verificar límites
        if (nueva_x < 0 or nueva_x >= self.tamaño or
                nueva_y < 0 or nueva_y >= self.tamaño):
            print("❌ Movimiento fuera de límites")
            return

        # Verificar si es transitable
        if self.grid[nueva_y, nueva_x] == 1:
            print("❌ No puedes atravesar una pared")
            return

        # Mover agente
        self.establecer_posicion_agente((nueva_x, nueva_y))
        self.imprimir_laberinto()

        # Verificar si llegó a alguna salida
        if (nueva_x, nueva_y) == self.salida_valida:
            print("\n🎉 ¡FELICIDADES! Encontraste la salida válida!")
        elif (nueva_x, nueva_y) in self.salidas_falsas:
            print(f"\n💀 ¡CUIDADO! Esta es una salida falsa. El juego continúa...")

    def mostrar_solucion(self, mostrar_camino: bool = False):
        """
        Muestra información sobre la solución del laberinto

        Args:
            mostrar_camino: Si mostrar el camino más corto a la salida válida
        """
        if not self.posicion_agente:
            print("❌ Posición del agente no establecida")
            return

        info = self.obtener_info_salidas()
        print(f"\n🔍 Información de la Solución:")
        print(f"   Salida válida: {info['salida_valida']}")
        print(f"   Salidas falsas: {info['salidas_falsas']}")
        print(f"   Distancia Manhattan: {info['distancia_a_salida_valida']}")

        if mostrar_camino:
            print("   (Función de pathfinding no implementada en esta versión)")
            print("   Pista: La salida válida está marcada con ✓ en amarillo")


# Ejemplo de uso completo
if __name__ == "__main__":
    print("🌟 Laberinto Dinámico Temporal con Sistema de Reparación - Versión Completa")
    print("=" * 80)

    # Crear laberinto con visualización automática y múltiples salidas
    laberinto = LaberintoDinamicoTemporal(
        tamaño=15,
        nodo_inicio=(1, 1),
        intervalo_base=1.5,
        auto_visualizar=True,  # Activar visualización automática
        num_salidas=4  # 4 salidas, solo 1 válida
    )

    # Generar laberinto usando algoritmo de Kruskal
    print("🔧 Generando laberinto inicial con algoritmo de Kruskal...")
    laberinto.generar_completamente()

    # Establecer posición del agente
    laberinto.establecer_posicion_agente((1, 1))

    print("\n🎯 Laberinto inicial con sistema de salidas:")
    laberinto.imprimir_laberinto()

    # Mostrar información de salidas
    info_salidas = laberinto.obtener_info_salidas()
    print(f"\n🚪 Sistema de Salidas:")
    print(f"   📊 Total de salidas: {info_salidas['total_salidas']}")
    print(f"   📍 Posiciones: {info_salidas['posiciones_salidas']}")
    print(f"   ✅ Salida válida: {info_salidas['salida_valida']} (¡secreta!)")
    print(f"   ❌ Salidas falsas: {info_salidas['salidas_falsas']}")
    print(f"   📏 Distancia a salida válida: {info_salidas['distancia_a_salida_valida']}")

    print(f"\n🎨 Leyenda de símbolos:")
    print(f"   A  = Agente (verde)")
    print(f"   ✓  = Salida válida (amarillo) - ¡Encuentra esta!")
    print(f"   ✗  = Salidas falsas (magenta) - ¡Evita estas!")
    print(f"   ▓  = Caminos críticos protegidos (cyan)")
    print(f"   █  = Paredes")
    print(f"   Espacios vacíos = Caminos transitables")
    print(f"   🔧 Sistema de reparación automática: ACTIVADO")

    # Mostrar información inicial
    peso_promedio = np.mean(laberinto.pesos_probabilidad[1:-1, 1:-1])
    print(f"\n📊 Estadísticas del laberinto:")
    print(f"   🎯 Peso promedio de inestabilidad: {peso_promedio:.3f}")
    print(f"   ⏱️  Intervalo base: {laberinto.intervalo_base}s")
    print(f"   🔧 Intervalo calculado: {laberinto._obtener_intervalo_actual():.2f}s")
    print(f"   🛡️  Caminos críticos protegidos: {len(laberinto.caminos_criticos)}")

    # Menú de opciones
    print(f"\n🎮 ¿Qué te gustaría hacer?")
    print("1. Demostración automática (15 segundos)")
    print("2. Modo interactivo completo")
    print("3. Simulación manual paso a paso")
    print("4. Prueba de reparación automática")

    try:
        opcion = input("\nSelecciona una opción (1-4): ").strip()

        if opcion == "1":
            # Demostración automática
            print("\n🚀 Iniciando demostración automática por 15 segundos...")
            print("   Los cambios se mostrarán automáticamente")
            print("   Observa cómo el sistema repara automáticamente los caminos")
            print("   Presiona Ctrl+C para interrumpir")

            laberinto.iniciar_actualizacion_temporal()
            time.sleep(15)
            laberinto.detener_actualizacion_temporal()

        elif opcion == "2":
            # Modo interactivo
            print("\n🎮 Iniciando modo interactivo...")
            print("   Podrás controlar el agente, activar/desactivar reparación, y más")
            laberinto.modo_interactivo()

        elif opcion == "3":
            # Simulación manual
            print("\n⚙️ Simulación manual paso a paso...")
            print("   Cada paso simula tiempo transcurrido con reparación automática")
            tiempo_simulado = 0
            paso_tiempo = 1.0  # 1 segundo por paso

            for i in range(20):  # 20 pasos
                print(f"\n--- Paso {i + 1}/20 (Tiempo: {tiempo_simulado:.1f}s) ---")
                cambios = laberinto.actualizar_manual_por_tiempo(paso_tiempo, mostrar_resultado=False)
                tiempo_simulado += paso_tiempo

                if cambios > 0:
                    print(f"🔄 {cambios} cambios aplicados")
                    laberinto.imprimir_laberinto()

                    # Mostrar reparaciones si las hubo
                    stats_rep = laberinto.obtener_estadisticas_reparacion()
                    if (stats_rep['ultima_reparacion'] and
                            stats_rep['ultima_reparacion']['timestamp'] > time.time() - paso_tiempo):
                        print(f"🔧 Reparación automática activada")
                else:
                    print("⏸️  Sin cambios en este intervalo")

                if i % 3 == 0:  # Mostrar info cada 3 pasos
                    info = laberinto.obtener_info_salidas()
                    print(f"📍 Agente en: {laberinto.posicion_agente}")
                    print(f"🎯 Distancia a salida válida: {info['distancia_a_salida_valida']}")

                time.sleep(0.5)  # Pausa para visualización

        elif opcion == "4":
            # Prueba de reparación
            print("\n🧪 Prueba del sistema de reparación automática...")
            print("   Se simularán bloqueos para probar la reparación")

            # Crear algunos bloqueos artificiales
            print("\n🔒 Creando bloqueos artificiales...")
            original_grid = laberinto.grid.copy()

            # Bloquear algunas rutas alrededor del agente
            x, y = laberinto.posicion_agente
            for dx, dy in [(1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < laberinto.tamaño and 0 <= ny < laberinto.tamaño:
                    laberinto.grid[ny, nx] = 1

            print("🔒 Bloqueos creados. Verificando reparación...")
            laberinto.imprimir_laberinto()

            # Forzar verificación de conectividad
            laberinto._verificar_y_reparar_conectividad()

            print("\n✅ Después de la reparación automática:")
            laberinto.imprimir_laberinto()

            # Mostrar estadísticas de reparación
            stats_rep = laberinto.obtener_estadisticas_reparacion()
            print(f"\n📊 Estadísticas de la prueba:")
            for k, v in stats_rep.items():
                print(f"   {k}: {v}")

        # Estadísticas finales
        stats = laberinto.obtener_estadisticas_temporales()
        info_final = laberinto.obtener_info_salidas()
        stats_reparacion = laberinto.obtener_estadisticas_reparacion()

        print(f"\n📈 Resumen final:")
        print(f"   🔢 Total de actualizaciones: {stats['total_actualizaciones']}")
        print(f"   🔄 Cambios totales: {stats['cambios_totales']}")
        print(f"   ⏱️  Tiempo total transcurrido: {stats['tiempo_acumulado']:.1f}s")
        print(f"   📊 Peso promedio final: {stats['peso_promedio_actual']:.3f}")
        print(f"   🎯 Victoria lograda: {'Sí' if info_final['agente_gano'] else 'No'}")
        print(f"   🔧 Reparaciones automáticas: {stats_reparacion['reparaciones_totales']}")
        print(f"   🛡️  Caminos críticos protegidos: {stats_reparacion['caminos_criticos']}")

        if not info_final['agente_gano']:
            print(f"\n💡 Pista: La salida válida estaba en {info_final['salida_valida']}")
            print("   ¡El sistema de reparación mantuvo los caminos abiertos!")

    except KeyboardInterrupt:
        laberinto.detener_actualizacion_temporal()
        print("\n👋 Programa interrumpido")
    except Exception as e:
        laberinto.detener_actualizacion_temporal()
        print(f"\n❌ Error: {e}")

    print("\n🎮 ¡Gracias por probar el Laberinto Dinámico con Reparación Automática!")
    print("🔗 Características implementadas:")
    print("   ✅ Algoritmo de Kruskal para generación inicial")
    print("   ✅ Sistema de k salidas (solo 1 válida)")
    print("   ✅ Dinamismo temporal basado en pesos probabilísticos")
    print("   ✅ Reparación automática de caminos bloqueados")
    print("   ✅ Visualización en tiempo real con colores")
    print("   ✅ Modo interactivo completo")
    print("   ✅ Thread-safe y configurable")