import pandas as pd
import matplotlib.pyplot as plt

# Archivos CSV
gen_file = "genetico.csv"
ucs_file = "ucs.csv"

# Leer datos
df_gen = pd.read_csv(gen_file)
df_ucs = pd.read_csv(ucs_file)

# Convertir tiempos de ns a segundos
df_gen["t_mean"] = df_gen["t_mean"] / 1e9
df_ucs["t_mean"] = df_ucs["t_mean"] / 1e9

df_gen["t_stdev"] = df_gen["t_stdev"] / 1e9
df_ucs["t_stdev"] = df_ucs["t_stdev"] / 1e9

# Agrupar por n y calcular promedios
gen_stats = df_gen.groupby("n").agg(
    mean_time=("t_mean", "mean"),
    std_time=("t_stdev", "mean"),
    success_rate=("success_rate", "mean")
).reset_index()

ucs_stats = df_ucs.groupby("n").agg(
    mean_time=("t_mean", "mean"),
    std_time=("t_stdev", "mean"),
    success_rate=("success_rate", "mean")
).reset_index()

# ---- Gráfico 1: tiempo medio ----
plt.figure(figsize=(8,5))
plt.errorbar(gen_stats["n"], gen_stats["mean_time"], yerr=gen_stats["std_time"],
             fmt="o-", capsize=5, label="Genético")
plt.errorbar(ucs_stats["n"], ucs_stats["mean_time"], yerr=ucs_stats["std_time"],
             fmt="s-", capsize=5, label="UCS")

plt.xlabel("Tamaño del laberinto (n)")
plt.ylabel("Tiempo medio (segundos)")
plt.title("Tiempo medio de resolución")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Gráfico 2: tasa de éxito ----
plt.figure(figsize=(8,5))
plt.plot(gen_stats["n"], gen_stats["success_rate"], marker="o", label="Genético")
plt.plot(ucs_stats["n"], ucs_stats["success_rate"], marker="s", label="UCS")
plt.xlabel("Tamaño del laberinto (n)")
plt.ylabel("Tasa de éxito")
plt.title("Comparación de éxito")
plt.legend()
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()
