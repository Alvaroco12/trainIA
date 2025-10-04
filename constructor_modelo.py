import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model

class ConstructorModelo:
    def __init__(self, dimension_entrada):
        self.dimension_entrada = dimension_entrada
        self.modelo = None

    def construir(self):
        tf.compat.v1.reset_default_graph()

        capa_entrada = Input(shape=(self.dimension_entrada,))
        capa_oculta1 = Dense(self.dimension_entrada, activation="relu")(capa_entrada)
        capa_oculta2 = Dense(64, activation="relu")(capa_oculta1)
        capa_salida = Dense(10, activation="softmax")(capa_oculta2)

        self.modelo = Model(inputs=capa_entrada, outputs=capa_salida)
        self.modelo.compile(loss="categorical_crossentropy",
                            optimizer="adam",
                            metrics=["accuracy"])
        return self.modelo

    def entrenar(self, trainX, trainY, testX, testY, epochs=5, batch_size=128):
        return self.modelo.fit(trainX, trainY,
                               validation_data=(testX, testY),
                               epochs=epochs,
                               batch_size=batch_size,
                               verbose=2)
