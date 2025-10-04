import matplotlib.pyplot as plt

class Visualizador:
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

    def mostrar_digito(self, indice):
        etiqueta = self.trainY[indice]
        imagen = self.trainX[indice].reshape([28, 28])
        plt.title(f"Datos de entrenamiento, índice: {indice}, Etiqueta: {etiqueta}")
        plt.imshow(imagen, cmap="gray_r")
        plt.show()
