from cargador_datos import CargadorDatos
from constructor_modelo import ConstructorModelo
from evaluador import Evaluador
from PIL import Image
import numpy as np

def main():
    # 1. Cargar y preparar datos
    cargador = CargadorDatos()
    trainX, trainY, testX, testY = cargador.preprocesar()

    # 2. Construir y entrenar modelo
    constructor = ConstructorModelo(cargador.dimension_entrada)
    modelo = constructor.construir()
    modelo.summary()
    constructor.entrenar(trainX, trainY, testX, testY, epochs=5)

    # 3. Evaluar modelo
    evaluador = Evaluador(modelo)
    evaluador.evaluar(testX, testY)

    # 4. Probar imagen externa
    ruta_imagen = "mi_numero.png"  # Cambia por la ruta de tu imagen
    numero_predicho = predecir_imagen(modelo, ruta_imagen, cargador.dimension_entrada)
    print(f"Número predicho para la imagen '{ruta_imagen}': {numero_predicho}")

def predecir_imagen(modelo, ruta, dimension_entrada):

    #Recibe el modelo y la ruta de una imagen de dígito.Devuelve el número predicho.
    imagen = Image.open(ruta).convert("L")
    imagen = imagen.resize((28, 28))
    array = np.array(imagen) / 255.0
    array = 1 - array
    array = array.reshape(1, dimension_entrada)
    pred = modelo.predict(array)
    return np.argmax(pred)

if __name__ == "__main__":
    main()
