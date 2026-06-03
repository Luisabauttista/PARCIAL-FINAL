# NeuralX
NeuralX es un entorno de desarrollo completo y educativo diseñado para el aprendizaje automático (Machine Learning), implementado totalmente desde cero en Python, sin depender de bibliotecas externas.
Este proyecto abarca desde el álgebra lineal básica y la gestión de archivos hasta la implementación de redes neuronales profundas, sirviendo como una herramienta integral para entender la lógica detrás de los modelos modernos.

# Características Principales

Matemáticas desde Cero (nx_math.py): Implementación de funciones trigonométricas, logarítmicas y exponenciales base.

Estructuras de Datos (nx_estructuras.py): Gestión propia de listas y arreglos con algoritmos de ordenamiento integrados.

Álgebra Lineal (nx_matrix.py): Motor de matrices que soporta operaciones fundamentales: suma, resta, etc.

Aprendizaje Automático:
Regresiones: Modelos de Regresión Lineal y Logística.
Redes Neuronales: Perceptrón Multicapa (MLP) con optimizador Adam y soporte para funciones de activación (relu, sigmoid, tanh, softmax).

Clustering: Algoritmos KMeans (con inicialización KMeans++) y DBSCAN.

Entrada/Salida (nx_io.py): Manejo nativo de archivos .txt y .csv.
Visualización (nx_plot.py): Gráficos renderizables en consola (ASCII) o exportación directa a formato SVG.

# Instalación y Requisitos
NeuralX no requiere instalaciones complejas ni gestores de dependencias. Solo necesitas tener instalado Python 3

# Cómo se ejecuta?
NeuralX incluye un motor de ejecución basado en ANTLR (neurax_visitor.py) que permite interpretar scripts personalizados escritos para NeuralX.

# Ejemplos de Uso 

# Matrices 

from nx_matrix import Matriz

A = Matriz([[1, 2], [3, 4]])
B = Matriz([[5, 6], [7, 8]])

# Operaciones de álgebra lineal
suma = A.suma(B)
print(suma)

inv = A.inversa() # Cálculo de inversa

# Entrenamiento de una red neuronal 

from nx_perceptron import MLP

# Inicializar MLP con 2 capas ocultas de 8 neuronas
modelo = MLP(capas=[8, 8], tarea="clasificacion")
modelo.compilar(tasa_aprendizaje=0.01, epocas=500)

# Datos de entrenamiento (ejemplo XOR)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

modelo.entrenar(X, y)
print(f"Predicción para [1, 0]: {modelo.predecir([1, 0])}")

# Entrenamiento de datos (I/O) 

from nx_io import ArchivoCSV

# Cargar dataset sin librerías externas
data = ArchivoCSV("dataset.csv")
print(f"Filas totales: {data.total_filas()}")
edad = data.columna("edad")

# Estructura del Proyecto

neurax_visitor.py: Motor central; evalúa el árbol de sintaxis y ejecuta el flujo de trabajo.

nx_estructuras.py: Tipos de datos personalizados.

nx_matrix.py: Motor de cálculo matricial.

nx_regresion.py: Algoritmos de predicción estadística.

nx_perceptron.py & nx_rna.py: Modelos de redes neuronales y clustering.

nx_plot.py: Generador de visualizaciones.

nx_io.py: Gestor de lectura y escritura de archivos.
