import pandas as pd
import numpy as np
import seaborn as sns  # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2  # OpenCV
import glob
import itertools
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.layers import *
from sklearn.metrics import confusion_matrix

""" Se carga el modelo de red neuronal entrenado y los distintos datos de entrada y datos de salida guardados en formato 
'numpy' """
model = load_model('/home/avalderas/img_slides/clinical/image&data/distant metastasis/inference/models/model_image&data_metastasis')

test_data = np.load('/home/avalderas/img_slides/clinical/image&data/distant metastasis/inference/test data/normalized/test_data_normalized.npy')
test_image = np.load('/home/avalderas/img_slides/clinical/image&data/distant metastasis/inference/test data/normalized/test_image_normalized.npy')
test_labels_metastasis = np.load('/home/avalderas/img_slides/clinical/image&data/distant metastasis/inference/test data/normalized/test_labels_normalized.npy')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate([test_data, test_image], test_labels_metastasis, verbose = 0)

""" -------------------------------------------------------------------------------------------------------------------
------------------------------------------- SECCIÓN DE EVALUACIÓN  ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
print("\n'Loss' de metástasis a distancia en el conjunto de prueba: {:.2f}\n""Sensibilidad de metástasis a distancia en "
      "el conjunto de prueba: {:.2f}%\n""Precisión de metástasis a distancia en el conjunto de prueba: {:.2f}%\n"
      "Especificidad de metástasis a distancia en el conjunto de prueba: {:.2f}% \n""Exactitud de metástasis a distancia "
      "en el conjunto de prueba: {:.2f}%\n""AUC-ROC de metástasis a distancia en el conjunto de prueba: {:.2f}\nAUC-PR "
      "de metástasis a distancia en el conjunto de "
      "prueba: {:.2f}".format(results[0], results[5] * 100, results[6] * 100, (results[3]/(results[3]+results[2])) * 100,
                              results[7] * 100, results[8], results[9]))

if results[5] > 0 or results[6] > 0:
    print("Valor-F de metástasis a distancia en el conjunto de "
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
        #print("Normalized confusion matrix")
    else:
        cm=cm
        #print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for il, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, il, cm[il, j], horizontalalignment = "center", color = "white" if cm[il, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Clase verdadera')
    plt.xlabel('Predicción')

# Recidivas
y_true_metastasis = test_labels_metastasis
y_pred_metastasis = np.round(model.predict([test_data, test_image]))

matrix_metastasis = confusion_matrix(y_true_metastasis, y_pred_metastasis, labels = [0, 1])
matrix_metastasis_classes = ['Sin metástasis distante', 'Con metástasis distante']

plot_confusion_matrix(matrix_metastasis, classes = matrix_metastasis_classes, title ='Matriz de confusión de metástasis '
                                                                                     'a distancia')
plt.show()

""" Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases.
Para implementarlas, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva: """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve

y_pred_prob_metastasis = model.predict([test_data, test_image]).ravel()
fpr, tpr, thresholds = roc_curve(test_labels_metastasis, y_pred_prob_metastasis)
auc_roc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for distant metastasis prediction')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision, recall, threshold_metastasis = precision_recall_curve(test_labels_metastasis, y_pred_prob_metastasis)
auc_pr = auc(recall, precision)

plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for distant metastasis prediction')
plt.legend(loc = 'best')
plt.show()