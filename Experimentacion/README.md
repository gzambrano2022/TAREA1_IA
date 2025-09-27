# uhr
A generic, C++, time performance tester.

## Helpers para realizar experimentos

Cada vez que queremos validar una hipótesis y comparar algoritmos se deben
realizar experimentos. Estos consumen tiempo y como es una tarea
repetitiva les ofrecemos soluciones precisas y estándares para estas.

En [uhr](./uhr.cpp) (reloj, en alemán) está un cronómetro configurable
que reporta _tiempo promedio, desviación estándar y cuartiles_ para un set de tests.

### Compilación

```bash
g++ -std=c++11 -O0 -o uhr uhr.cpp
```

Se pueden usar versiones más nuevas de C++ (14, 17, 20), pero **no** anteriores
a C++11. Se debe usar `-O0` para que el compilador no optimice el loop en
donde se realiza la medición de tiempo. Así que se recomienda programar
usando técnicas de optimización como [hoisting](https://en.wikipedia.org/wiki/Loop-invariant_code_motion)
para evitar recalcular en ciclos.

### Uso

```bash
./uhr <filename>.csv <runs> <lower> <upper> <step>
```

Donde `<filename>` es el nombre de archivo donde se escribirán los resultados.
El archivo **no** debe existir previamente y debe tener extensión `.csv`. Los
siguientes argumentos describen las pruebas, todos deben ser positivos e
indican cuántas repeticiones por prueba, límites inferior y superior y
el salto prueba a prueba, respectivamente.

`<runs>` debe ser al menos igual a 32, esto para poder tener
los caches en caliente y poder tener suficientes muestras para confianza
estadística. Yo tiendo a hacer 256 repeticiones si es que es permisivo esto
en tiempo, en caso contrario uso 64 y si es que cada experimento se demora
alrededor de un segundo es que uso 32.

Se recomienda que se deben escoger unidades de tiempo que sean significativas:
es decir si todo se ejecuta rápidamente, usar nanosegundos, si se demora un
poco más, usar micro segundos, luego milisegundos... En este sentido es bueno
tener nombres de archivos _semánticos_. Yo uso: `<elemento probado> + '_' + <unidad
de tiempo>`. Así los CSVs se describen solos, se puede saber qué información
contienen y en qué unidades está.

### Gráficos

Se recomienda usar [matplotlib](https://matplotlib.org) para hacer los gráficos.

Una alternativa es [pgfplots](https://ctan.org/pkg/pgfplots?lang=en) que sirve
para crear un gráfico en LaTeX directamente de un CSV.

### Tablas

En [csvltx](https://github.com/leonardlover/csvltx) hay una herramienta que
ayuda al parsing de archivos CSV en formato de tabla para uso en LaTeX.

Para poder ejecutarla, deben tener [Rust](https://www.rust-lang.org) instalado,
en particular `cargo` y simplemente deben clonar el repositorio y hacer
```bash
cargo run <filename> <unit>
```

Donde `<filename>` es el archivo CSV que fue escrito por uhr y `<unit>` es
la unidad de los datos del CSV. Esto imprimirá en la consola el texto que
hay que copiar en el archivo LaTeX para mostrar la tabla.

Tambien existe una alternativa en Python usando Pandas, que es
[`to_latex`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html),
el uso de este requiere cargar el CSV como DataFrame y usar el paquete `booktabs`
en LaTeX, así que la alternativa en Rust funciona puramente en la terminal y no
requiere adiciones a LaTeX.

### Ejemplos

En [plotter](./plotter.py) se muestra cómo crear gráficos usando matplotlib.
Ver ejemplos de estos en [sample_plots](./sample_plots.png).

En [experimental_data](./experimental_data) se encuentran algunos CSVs generados con
uhr (con nombres semánticos) y también se presenta un [ejemplo](./experimental_data/ejemplo/ejemplo.pdf) para ver qué tal
quedan los gráficos usando pgfplots y las tablas usando csvltx.
[Aquí](./experimental_data/ejemplo/ejemplo.tex) se puede ver la fuente de este ejemplo.
