import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix # Para realizar la matriz de confusión
import itertools

""" Se carga el modelo de red neuronal entrenado y los distintos datos de entrada y datos de salida guardados en formato 
'numpy' """
model = load_model('/home/avalderas/img_slides/correlations/anatomopathologic-clinical/inference/test_data&models/anatomopathologic-clinical.h5')

test_tabular_data = np.load('/home/avalderas/img_slides/correlations/anatomopathologic-clinical/inference/test_data&models/test_data.npy')

test_labels_metastasis = np.load('/home/avalderas/img_slides/correlations/anatomopathologic-clinical/inference/test_data&models/test_labels_metastasis.npy')
test_labels_survival = np.load('/home/avalderas/img_slides/correlations/anatomopathologic-clinical/inference/test_data&models/test_labels_survival.npy')
test_labels_relapse = np.load('/home/avalderas/img_slides/correlations/anatomopathologic-clinical/inference/test_data&models/test_labels_relapse.npy')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_tabular_data, [test_labels_metastasis, test_labels_survival, test_labels_relapse],
                         verbose = 0)

print("\n'Loss' de la metástasis a distancia en el conjunto de prueba: {:.2f}\n""Sensibilidad de la metástasis a "
      "distancia en el conjunto de prueba: {:.2f}\n""Precisión de la metástasis a distancia en el conjunto de prueba: "
      "{:.2f}\n""Especifidad de la metástasis a distancia en el conjunto de prueba: {:.2f} \n""Exactitud de la "
      "metástasis a distancia en el conjunto de prueba: {:.2f} %\n""AUC-ROC de la metástasis a distancia en el conjunto "
      "de prueba: {:.2f}".format(results[1], results[8], results[9], results[6]/(results[6]+results[5]),
                                 results[10] * 100, results[11]))

if results[8] > 0 or results[9] > 0:
    print("Valor-F de la metástasis a distancia en el conjunto de prueba: {:.2f}".format((2 * results[8] * results[9]) /
                                                                                         (results[8] + results[9])))

print("\n'Loss' de la supervivencia en el conjunto de prueba: {:.2f}\n""Sensibilidad de la supervivencia en el conjunto "
      "de prueba: {:.2f}\n""Precisión de la supervivencia en el conjunto de prueba: {:.2f}\n""Especifidad de la "
      "supervivencia en el conjunto de prueba: {:.2f} \n""Exactitud de la supervivencia en el conjunto de prueba: {:.2f} "
      "%\n""AUC-ROC de la supervivencia en el conjunto de prueba: {:.2f}".format(results[2], results[16], results[17],
                                                                                 results[14]/(results[14]+results[13]),
                                                                                 results[18] * 100, results[19]))
if results[16] > 0 or results[17] > 0:
    print("Valor-F de la supervivencia en el conjunto de prueba: {:.2f}".format((2 * results[16] * results[17]) /
                                                                                (results[16] + results[17])))

print("\n'Loss' de recidivas en el conjunto de prueba: {:.2f}\n""Sensibilidad de recidivas en el conjunto de prueba: "
      "{:.2f}\n""Precisión de recidivas en el conjunto de prueba: {:.2f}\n""Especifidad de recidivas en el conjunto de "
      "prueba: {:.2f} \n""Exactitud de recidivas en el conjunto de prueba: {:.2f} %\n""AUC-ROC de recidivas en el "
      "conjunto de prueba: {:.2f}".format(results[3], results[24], results[25], results[22]/(results[22]+results[21]),
                                                                  results[26] * 100, results[27]))
if results[24] > 0 or results[25] > 0:
    print("Valor-F de recidivas en el conjunto de prueba: {:.2f}".format((2 * results[24] * results[25]) /
                                                                         (results[24] + results[25])))

""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos. 
Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión', cmap = plt.cm.Blues):
    """ Imprime y dibuja la matriz de confusión. Se puede normalizar escribiendo el parámetro `normalize=True`. """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(2)
        #print("Normalized confusion matrix")
    else:
        cm=cm
        #print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for il, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, il, cm[il, j], horizontalalignment="center", color="white" if cm[il, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Clase verdadera')
    plt.xlabel('Predicción')

# Metástasis a distancia
y_true_metastasis = test_labels_metastasis
y_pred_metastasis = np.round(model.predict(test_tabular_data)[0])

matrix_metastasis = confusion_matrix(y_true_metastasis, y_pred_metastasis, labels = [0, 1])
matrix_metastasis_classes = ['Sin Metástasis', 'Con Metástasis']

plot_confusion_matrix(matrix_metastasis, classes = matrix_metastasis_classes, title = 'Matriz de confusión de presencia'
                                                                                      ' de metástasis a distancia')
plt.show()

# Supervivencia
y_true_survival = test_labels_survival
y_pred_survival = np.round(model.predict(test_tabular_data)[1])

matrix_survival = confusion_matrix(y_true_survival, y_pred_survival, labels = [0, 1])
matrix_survival_classes = ['Viviendo', 'Fallecida']

plot_confusion_matrix(matrix_survival, classes = matrix_survival_classes, title = 'Matriz de confusión de supervivencia'
                                                                                  ' del paciente')
plt.show()

# Recidivas
y_true_relapse = test_labels_relapse
y_pred_relapse = np.round(model.predict(test_tabular_data)[2])

matrix_relapse = confusion_matrix(y_true_relapse, y_pred_relapse, labels = [0, 1])
matrix_relapse_classes = ['Sin Recidiva', 'Con Recidiva']

plot_confusion_matrix(matrix_relapse, classes = matrix_relapse_classes, title = 'Matriz de confusión de recidivas del '
                                                                                ' paciente')
plt.show()