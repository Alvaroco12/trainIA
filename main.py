# main.py
from cargador_datos import CargadorDatos
from visualizador import Visualizador
from constructor_modelo import ConstructorModelo
from evaluador import Evaluador

from PIL import Image
import numpy as np

def preprocesar_imagen(ruta_imagen):
    img = Image.open(ruta_imagen).convert('L')  # Escala de grises
    img = img.resize((28, 28))                  # Redimensionar a 28x28
    img_array = np.array(img)
    
    # Invertir colores si es necesario (MNIST es blanco sobre negro)
    img_array = 255 - img_array
    
    # Normalizar y aplanar
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28*28)
    return img_array

def main():
    # 1. Cargar y preparar datos
    cargador = CargadorDatos()
    trainX, trainY, testX, testY = cargador.preprocesar()

    # 2. Visualizar un ejemplo de MNIST
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

    # 5. Predecir número externo
    ruta_imagen = "mi_numero.png"  # Cambia por la ruta de tu imagen
    imagen = preprocesar_imagen(ruta_imagen)
    prediccion = modelo.predict(imagen)
    numero = np.argmax(prediccion)
    print("\nNúmero predicho de la imagen externa:", numero)

if __name__ == "__main__":
    main()
