import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class CargadorDatos:
    def __init__(self):
        (self.trainX, self.trainY), (self.testX, self.testY) = mnist.load_data()
        self.dimension_entrada = self.trainX.shape[1] * self.trainX.shape[2]

    def preprocesar(self):
        trainX_norm = self.trainX.astype('float32') / 255.0
        testX_norm = self.testX.astype('float32') / 255.0

        trainX_norm = trainX_norm.reshape(trainX_norm.shape[0], self.dimension_entrada)
        testX_norm = testX_norm.reshape(testX_norm.shape[0], self.dimension_entrada)

        trainY_cate = to_categorical(self.trainY, num_classes=10)
        testY_cate = to_categorical(self.testY, num_classes=10)

        return trainX_norm, trainY_cate, testX_norm, testY_cate

