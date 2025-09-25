from maze import Laberinto
from agent_ucsm import A_UCS
import time

if __name__ == "__main__":

    # Inicializar laberinto
    laberinto = Laberinto(
        tamaño=15,
        nodo_inicio=(1, 1),
        intervalo_base=10.0,
        auto_visualizar=False,
        num_salidas=3
    )
    laberinto.generar_completamente()

    # Inicializar agente
    agente = A_UCS(laberinto, 1,1)
    laberinto.imprimir_laberinto()

    # Iniciar simulacion
    laberinto.iniciar_actualizacion_temporal()

    max_pasos = 500

    try:
        while agente.pasos < max_pasos and not agente.meta_alcanzada():
            nueva_pos = agente.mover()

            if nueva_pos is None:
                print("Sin rutas disponibles")
                break
            # Nueva posicion y reimpresion del laberinto en terminal
            laberinto.establecer_posicion_agente(nueva_pos)
            print("\n")
            laberinto.imprimir_laberinto()
            time.sleep(1.0)

        # Resultados
        if agente.meta_alcanzada():
            print("EXITO: Agente encontro la salida valida")
        else:
            print("FALLO: Agente no encontro la salida")

        laberinto.imprimir_laberinto()

    finally:
        laberinto.detener_actualizacion_temporal()