from laberinto import Laberinto
from agent_ucsm import A_UCS
import random
import sys
import time

if __name__ == "__main__":

    tamaño = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    salidas = tamaño // 3

    # Inicializar laberinto
    laberinto = Laberinto(
        tamaño=tamaño,
        x_i=1,
        y_i=1,
        intervalo=10.0,
        num_salidas= salidas
    )
    laberinto.generar_completamente()

    # Inicializar agente
    agente = A_UCS(laberinto, 1,1)
    laberinto.imprimir_laberinto()

    # Iniciar simulacion
    laberinto.iniciar_actualizacion_temporal()

    max_pasos = 500
    max_tiempo = 60.0  # segundos
    start_time = time.time()

    while agente.pasos < max_pasos and not agente.meta_alcanzada():

        if time.time() - start_time > max_tiempo:
            print("FALLO: Tiempo máximo excedido")
            exit(1)  # Fallo

        nueva_pos = agente.mover()

        if nueva_pos is None:
            print("Sin rutas disponibles")
            break
        # Nueva posicion y reimpresion del laberinto en terminal
        laberinto.establecer_posicion_agente(nueva_pos[0],nueva_pos[1])
        print("\n")
        laberinto.imprimir_laberinto()

    # Resultados
    if agente.meta_alcanzada():
        print("EXITO: Agente encontro la salida valida")
        exit(0)  # Exito
    else:
        print("FALLO: Agente no encontro la salida")
        exit(1)  # Fallo
