https://github.com/Alvaroco12/trainIA.git

# Reconocimiento de Dígitos con IA

Este proyecto implementa una red neuronal para reconocer dígitos escritos a mano utilizando el dataset MNIST. Está modularizado en clases para manejo de datos, construcción y entrenamiento del modelo, evaluación y prueba con imágenes externas.

---

## Versión de Python

- **Python 3.10.11**

---

## Estructura del proyecto

| Archivo | Descripción |
|---------|-------------|
| `main.py` | Archivo principal que carga los datos, construye y entrena la red neuronal, evalúa el modelo y permite probar imágenes externas. |
| `cargador_datos.py` | Clase `CargadorDatos` para cargar y preprocesar el dataset MNIST. Normaliza los datos y convierte etiquetas a one-hot. |
| `constructor_modelo.py` | Clase `ConstructorModelo` que define la arquitectura de la red neuronal, la compila y la entrena. |
| `evaluador.py` | Clase `Evaluador` que calcula precisión, matriz de confusión y reporte de clasificación. |
| `prueba.py` | Permite probar una imagen externa (`mi_numero.png`) y ver qué número predice la IA. Convierte la imagen a escala de grises, la normaliza y la redimensiona a 28x28. |

---

## Preparar la imagen de prueba

Para probar la IA con un número hecho a mano:

1. Abre **Paint** y crea un nuevo archivo de **100x100 px**.
2. Selecciona el **pincel de 3 px** y dibuja el número **negro**, lo más centrado posible.
3. Guarda la imagen con el nombre `mi_numero.png` en la carpeta del proyecto o indica su ruta en `prueba.py`.

> La red neuronal espera números centrados y con contraste alto, para que coincidan con el formato MNIST.

---

## Cómo usar

1. Clonar el repositorio:

git clone https://github.com/Alvaroco12/trainIA.git

2. Crear y activar un entorno virtual:

py -3.10 -m venv .venv

.venv\Scripts\activate  # Windows


python -m venv .venv

source .venv/bin/activate  # macOS / Linux

3. Instalar dependencias:

pip install -r requirements.txt

4. Ejecutar main.py para entrenar y evaluar el modelo:

python main.py

## Notas importantes:

Asegúrate de que la imagen siga las especificaciones de tamaño y pincel indicadas.

La red neuronal funciona mejor con imágenes centradas y números de alto contraste.
