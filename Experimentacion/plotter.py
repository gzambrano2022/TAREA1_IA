import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("resultados.csv")

t_mean_sec = data["t_mean"] / 1e9
t_stdev_sec = data["t_stdev"] / 1e9

plt.errorbar(data["n"], t_mean_sec, t_stdev_sec, linestyle="None", marker='.', ecolor='tab:red')
plt.xlabel("n")
plt.ylabel("Tiempo medio [s]")
plt.title("Rendimiento Agente UCS")
plt.grid(True)
plt.show()

# Preparar datos para boxplot
quartiles_data = [data["t_Q0"]/1e9, data["t_Q1"]/1e9,
                  data["t_Q2"]/1e9, data["t_Q3"]/1e9, data["t_Q4"]/1e9]
labels = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4']

plt.figure(figsize=(10, 6))
plt.boxplot(quartiles_data, labels=labels)
plt.ylabel("Tiempo [s]")
plt.title("Distribución de Tiempos por Cuartiles")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(data["n"], data["t_Q0"]/1e9, 'o-', label='Q0 (mínimo)')
plt.plot(data["n"], data["t_Q1"]/1e9, 'o-', label='Q1 (25%)')
plt.plot(data["n"], data["t_Q2"]/1e9, 'o-', label='Q2 (mediana)')
plt.plot(data["n"], data["t_Q3"]/1e9, 'o-', label='Q3 (75%)')
plt.plot(data["n"], data["t_Q4"]/1e9, 'o-', label='Q4 (máximo)')
plt.plot(data["n"], t_mean_sec, 'k--', label='Media')

plt.xlabel("n")
plt.ylabel("Tiempo [s]")
plt.title("Evolución de Cuartiles vs n")
plt.legend()
plt.grid(True)
plt.show()