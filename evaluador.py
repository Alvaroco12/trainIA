import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class Evaluador:
    def __init__(self, modelo):
        self.modelo = modelo

    def evaluar(self, testX, testY):
        predicciones = np.array(self.modelo.predict(testX)).argmax(axis=1)
        reales = testY.argmax(axis=1)
        precision = np.mean(predicciones == reales)

        print("Precisión en test:", precision)
        print("Matriz de confusión:")
        print(confusion_matrix(y_true=reales, y_pred=predicciones))
        print("Reporte de clasificación:")
        print(classification_report(reales, predicciones))
