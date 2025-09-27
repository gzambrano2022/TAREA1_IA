from laberinto import Laberinto
from agent_genet import A_GENET
import time
import random

if __name__ == "__main__":

    # Inicializar laberinto
    laberinto = Laberinto(
        tamaño=10,
        x_i=1,
        y_i=1,
        intervalo=10.0,
        num_salidas=random.randint(1, 100)
    )
    laberinto.generar_completamente()

    # Inicializar agente genetico
    agente = A_GENET(laberinto, 1, 1)
    laberinto.imprimir_laberinto()

    # Iniciar simulacien
    laberinto.iniciar_actualizacion_temporal()

    try:
        # Obtener la mejor solucion
        mejor_cromosoma = agente.evolucionar()

        x, y = 1, 1
        pasos = 0
        exito = False

        # Recorrer el cromosoma como si fueran pasos del agente
        for movimiento in mejor_cromosoma:

            dx, dy = movimiento
            nx, ny = x + dx, y + dy

            # Validar limites y paredes dinamicas
            if (0 <= nx < laberinto.tamaño and 0 <= ny < laberinto.tamaño and laberinto.grid[ny, nx] == 0):
                x, y = nx, ny
                pasos += 1

            # Actualizar posición del agente en el laberinto
            laberinto.establecer_posicion_agente(x, y)
            print("\n")
            laberinto.imprimir_laberinto()

            # Verificar si llegó a una salida
            if (x, y) == laberinto.salida_valida:
                exito = True
                break
            elif (x, y) in laberinto.salidas_falsas:
                print("FALLO: Agente llegó a una salida falsa")
                break

        # Resultados
        if exito:
            print(f"EXITO: Agente genético encontró la salida en {pasos} pasos")
        else:
            print("FALLO: Agente genético no encontró la salida")

        laberinto.imprimir_laberinto()

    finally:
        laberinto.detener_actualizacion_temporal()
