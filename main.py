from cargador_datos import CargadorDatos
from visualizador import Visualizador
from constructor_modelo import ConstructorModelo
from evaluador import Evaluador

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

if __name__ == "__main__":
    main()
