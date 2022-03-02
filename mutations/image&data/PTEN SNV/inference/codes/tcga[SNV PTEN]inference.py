""" Librerías """
import pandas as pd
import numpy as np
import seaborn as sns  # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2  # OpenCV
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve, auc, precision_recall_curve

""" Se carga el modelo de red neuronal entrenado y los distintos datos de entrada y datos de salida guardados en formato 
'numpy' """
model = load_model('/home/avalderas/img_slides/mutations/image&data/PTEN SNV/inference/models/model_image&data_PTEN_01_0.69.h5')

test_data = np.load('/home/avalderas/img_slides/mutations/image&data/PTEN SNV/inference/test data/normalized/test_data.npy')
test_image = np.load('/home/avalderas/img_slides/mutations/image&data/PTEN SNV/inference/test data/normalized/test_image.npy')
test_labels = np.load('/home/avalderas/img_slides/mutations/image&data/PTEN SNV/inference/test data/normalized/test_labels.npy')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate([test_data, test_image], test_labels, verbose = 0)

print("\n'Loss' de las mutaciones SNV del gen PTEN en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "SNV del gen PTEN en el conjunto de prueba: {:.2f}%\n""Precisión de las mutaciones SNV del gen PTEN en el "
      "conjunto de prueba: {:.2f}%\n""Especificidad de las mutaciones SNV del gen PTEN en el conjunto de prueba: "
      "{:.2f}% \n""Exactitud de las mutaciones SNV del gen FGF3 en el conjunto de prueba: {:.2f}%\n""AUC-ROC de las "
      "mutaciones SNV del gen PTEN en el conjunto de prueba: {:.2f}\nAUC-PR de las mutaciones SNV del gen PTEN en "
      "el conjunto de prueba: {:.2f}".format(results[0], results[5] * 100, results[6] * 100,
                                             (results[3]/(results[3]+results[2])) * 100, results[7] * 100, results[8],
                                             results[9]))

if results[5] > 0 or results[6] > 0:
    print("Valor-F de las mutaciones SNV del gen PTEN en el conjunto de "
          "prueba: {:.2f}".format((2 * results[5] * results[6]) / (results[5] + results[6])))

""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos.
Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Matriz de confusión', cmap = plt.cm.Blues):
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
    else:
        cm=cm

    thresh = cm.max() / 2.
    for il, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, il, cm[il, j], horizontalalignment = "center", color = "white" if cm[il, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Clase')
    plt.xlabel('Predicción')

# Matriz de confusión
labels = test_labels
predictions = np.round(model.predict([test_data, test_image]))

matrix = confusion_matrix(labels, predictions, labels = [0, 1])
matrix_classes = ['Sin mutación', 'Con mutación']

plot_confusion_matrix(matrix, classes = matrix_classes, title ='Matriz de confusión [SNV PTEN]')
plt.show()

""" Para terminar, se calculan las curvas ROC. """
predictions_raw = model.predict([test_data, test_image])

# AUC-ROC
fpr, tpr, thresholds_roc = roc_curve(labels, predictions_raw)
auc_roc = auc(fpr, tpr)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for SNV of PTEN gene')
plt.legend(loc='best')
plt.show()

# AUC-PR
precision, recall, threshold_pr = precision_recall_curve(labels, predictions_raw)
auc_pr = auc(recall, precision)
plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for SNV of PTEN gene')
plt.legend(loc='best')
plt.show()