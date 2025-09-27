from laberinto import Laberinto
from agent_genet import A_GENET

if __name__ == "__main__":

    # Inicializar laberinto
    laberinto = Laberinto(
        tamaño=10,
        x_i=1,
        y_i=1,
        intervalo=10.0,
        num_salidas=3
    )
    laberinto.generar_completamente()

    # Crear agente genético
    agente_gen = A_GENET(laberinto, xi=1, yi=1)

    # Imprimir laberinto inicial
    laberinto.establecer_posicion_agente(1,1)
    print("Laberinto inicial:")
    laberinto.imprimir_laberinto()

    # Ejecutar evolución
    print("\nEjecutando algoritmo genetico...")
    mejor_cromosoma = agente_gen.evolucionar()

    # Simular mejor cromosoma paso a paso
    x, y = 1, 1
    laberinto.establecer_posicion_agente(x, y)
    print("\nSimulación del mejor cromosoma:")
    laberinto.imprimir_laberinto()

    for movimiento in mejor_cromosoma:
        dx, dy = movimiento
        nx, ny = x + dx, y + dy

        # Validar límites y paredes
        if 0 <= nx < laberinto.tamaño and 0 <= ny < laberinto.tamaño and laberinto.grid[ny, nx] == 0:
            x, y = nx, ny

        laberinto.establecer_posicion_agente(x, y)
        print("\n")
        laberinto.imprimir_laberinto()

        # Terminar si alcanza salida válida
        if (x, y) == laberinto.salida_valida:
            print("\nEXITO: Algoritmo genético encontró la salida")
            break
        elif (x, y) in laberinto.salidas_falsas:
            print("\nFALLO: Llegó a una salida falsa")
            break
