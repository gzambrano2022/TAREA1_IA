import sys
import time
from laberinto import Laberinto
from agent_genet import A_GENET
import random

if __name__ == "__main__":

    tamaño = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    salidas = tamaño // 3
    cromosomas = tamaño*5
    generacion= tamaño*7

    # Inicializar laberinto
    laberinto = Laberinto(
        tamaño=tamaño,
        x_i=1,
        y_i=1,
        intervalo=10.0,
        num_salidas=salidas
    )
    laberinto.generar_completamente()

    # Inicializar agente genetico
    agente = A_GENET(laberinto, 1, 1,cromosomas,generacion)
    laberinto.imprimir_laberinto()

    # Iniciar simulacien
    laberinto.iniciar_actualizacion_temporal()

    # Obtener la mejor solucion
    mejor_cromosoma = agente.evolucionar()

    x, y = 1, 1
    pasos = 0
    exito = False
    max_tiempo = 60.0
    start_time = time.time()

    # Recorrer el cromosoma como si fueran pasos del agente
    for movimiento in mejor_cromosoma:

        if time.time() - start_time > max_tiempo:
            print("FALLO: Tiempo máximo excedido")
            exit(1)  # Fallo

        dx, dy = movimiento
        nx, ny = x + dx, y + dy

        # Validar limites y paredes dinamicas
        if (0 <= nx < laberinto.tamaño and 0 <= ny < laberinto.tamaño and laberinto.grid[ny, nx] == 0):
            x, y = nx, ny
            pasos += 1

        # Actualizar posicion del agente en el laberinto
        laberinto.establecer_posicion_agente(x, y)
        print("\n")
        laberinto.imprimir_laberinto()

        # Verificar si llego a una salida
        if (x, y) == laberinto.salida_valida:
            exito = True
            break
        elif (x, y) in laberinto.salidas_falsas:
            print("FALLO: Salida falsa")
            break

    # Resultados
    if exito:
        print(f"EXITO: Agente encontro la salida en {pasos} pasos")
        exit(0)  # Exito
    else:
        print("FALLO: Agente no encontro la salida")
        exit(1)  # Fallo
