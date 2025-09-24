# main_lrta.py
from maze import LaberintoDinamicoTemporal
from agent_lrta import AgenteLRTA
import time
import numpy as np

if __name__ == "__main__":
    laberinto = LaberintoDinamicoTemporal(
        tamaño=15,
        nodo_inicio=(1, 1),
        intervalo_base=1.5,
        auto_visualizar=True,
        num_salidas=4
    )
    laberinto.generar_completamente()
    laberinto.establecer_posicion_agente((1, 1))
    laberinto.imprimir_laberinto()

    # Crear agente LRTA* hacia la salida válida
    agente = AgenteLRTA(laberinto, inicio=(1, 1), objetivo=laberinto.salida_valida)

    print("\n🚀 Simulación del agente LRTA* hacia la salida válida...")

    while not agente.meta_alcanzada():
        nueva_pos = agente.mover()
        laberinto.establecer_posicion_agente(nueva_pos)
        laberinto.imprimir_laberinto()
        time.sleep(0.5)

    print("\n🎉 El agente alcanzó la salida válida!")
