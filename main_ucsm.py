from maze import Laberinto
from agent_ucsm import A_UCS
import time


def ejecutar_simulacion():
    """Ejecuta una simulación con UCS y mapeo dinámico"""

    # Crear laberinto
    laberinto = Laberinto(
        tamaño=25,
        nodo_inicio=(1, 1),
        intervalo_base=10.0,
        auto_visualizar=False,
        num_salidas=3
    )
    laberinto.generar_completamente()
    laberinto.establecer_posicion_agente((1, 1))

    # Crear agente
    agente = A_UCS(laberinto, inicio=(1, 1))

    print("Estado inicial del laberinto:")
    print(f"Salidas: {laberinto.salidas}")
    print(f"Salida válida: {laberinto.salida_valida}")
    laberinto.imprimir_laberinto()

    # Iniciar simulación
    laberinto.iniciar_actualizacion_temporal()

    inicio_tiempo = time.time()
    max_pasos = 500

    try:
        while agente.pasos < max_pasos and not agente.meta_alcanzada():
            nueva_pos = agente.mover()

            if nueva_pos is None:
                print("Agente sin rutas disponibles")
                break

            laberinto.establecer_posicion_agente(nueva_pos)

            # Mostrar cada paso
            print(f"\n--- Paso {agente.pasos} ---")
            print(f"Agente en: {nueva_pos}")
            print(f"Salidas restantes: {agente.salidas}")
            print(f"Pasos en ruta actual: {len(agente.ruta)}")
            laberinto.imprimir_laberinto()

            time.sleep(1.0)

        tiempo_total = time.time() - inicio_tiempo

        # Resultados
        print("\n=== RESULTADOS ===")
        if agente.meta_alcanzada():
            print("ÉXITO: Agente encontró la salida válida")
        else:
            print("FALLO: Agente no encontró la salida")

        print(f"Pasos totales: {agente.pasos}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")
        print(f"Posición final: {agente.pos}")
        print(f"Salidas exploradas: {3 - len(agente.salidas)}/3")

        laberinto.imprimir_laberinto()

        return {
            'exito': agente.meta_alcanzada(),
            'pasos': agente.pasos,
            'tiempo': tiempo_total
        }

    finally:
        laberinto.detener_actualizacion_temporal()


if __name__ == "__main__":
    print("Ejecutando simulación: UCS con Mapeo Dinámico...")
    resultado = ejecutar_simulacion()