from cargador_datos import CargadorDatos
from visualizador import Visualizador
from constructor_modelo import ConstructorModelo
from evaluador import Evaluador
from PIL import Image
import numpy as np

def main():
    # 1. Cargar y preparar datos
    cargador = CargadorDatos()
    trainX, trainY, testX, testY = cargador.preprocesar()

    # 2. Visualizar un ejemplo
    visualizador = Visualizador(cargador.trainX, cargador.trainY)
    visualizador.mostrar_digito(2)

    # 3. Construir y entrenar modelo
    constructor = ConstructorModelo(cargador.dimension_entrada)
    modelo = constructor.construir()
    modelo.summary()
    constructor.entrenar(trainX, trainY, testX, testY, epochs=5)

    # 4. Evaluar modelo
    evaluador = Evaluador(modelo)
    evaluador.evaluar(testX, testY)

    # 5. Probar imagen externa
    ruta_imagen = "mi_numero.png"  # Cambia por la ruta de tu imagen
    numero_predicho = predecir_imagen(modelo, ruta_imagen, cargador.dimension_entrada)
    print(f"Número predicho para la imagen '{ruta_imagen}': {numero_predicho}")

def predecir_imagen(modelo, ruta, dimension_entrada):
    """
    Recibe el modelo y la ruta de una imagen de dígito.
    Devuelve el número predicho.
    """
    # Abrir imagen y convertir a escala de grises
    imagen = Image.open(ruta).convert("L")
    # Redimensionar a 28x28 (MNIST)
    imagen = imagen.resize((28, 28))
    # Convertir a array y normalizar
    array = np.array(imagen) / 255.0
    # Invertir colores si es necesario (MNIST fondo negro, número blanco)
    array = 1 - array
    # Aplanar a vector
    array = array.reshape(1, dimension_entrada)
    # Predecir
    pred = modelo.predict(array)
    return np.argmax(pred)

if __name__ == "__main__":
    main()
