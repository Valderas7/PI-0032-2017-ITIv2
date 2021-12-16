""" Programa para evaluar las métricas del subconjunto de test (conjunto total de imágenes de TCGA) de datos
anatomopatológicos.

Table 2. Anatomical pathology samples in TCGA dataset with lymph node patients (552 patients)

|       Tumor Type      |
|   Type    |   Samples |
    IDC           402
    ILC           105
Mucinous            6
Medillary           1
Metaplastic         1
"""

""" Librerías """
import pandas as pd
import numpy as np
import seaborn as sns  # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2  # OpenCV
import glob
import tensorflow as tf
import itertools
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

""" Se carga el modelo de red neuronal entrenado y los distintos datos de entrada y datos de salida guardados en formato 
'numpy' """
model = load_model(
    '/anatomical_pathology_data/image/tumor_type_without_mixed/inference/models/model_image_tumor_type_06_0.47_ultimate.h5')

test_image_data = np.load('/home/avalderas/img_slides/anatomical_pathology_data/image/tumor_type_without_mixed/inference/test_data/test_image_ultimate.npy')
test_labels_tumor_type = np.load('/home/avalderas/img_slides/anatomical_pathology_data/image/tumor_type_without_mixed/inference/test_data/test_labels_tumor_type_ultimate.npy')

""" Una vez entrenado el modelo se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data, test_labels_tumor_type, verbose = 0)

print("\n'Loss' del tipo histológico en el conjunto de prueba: {:.2f}\n""Sensibilidad del tipo histológico en el "
      "conjunto de prueba: {:.2f}%\n""Precisión del tipo histológico en el conjunto de prueba: {:.2f}%\n""Especificidad "
      "del tipo histológico en el conjunto de prueba: {:.2f}% \n""Exactitud del tipo histológico en el conjunto de prueba: "
      "{:.2f}%\n""AUC-ROC del tipo histológico en el conjunto de prueba: {:.2f}\n""AUC-PR del tipo histológico en el "
      "conjunto de prueba: {:.2f}".format(results[0], results[5] * 100, results[6] * 100,
                                          (results[3]/(results[3]+results[2])) * 100, results[7] * 100, results[8],
                                          results[9]))
if results[5] > 0 or results[6] > 0:
    print("Valor-F del tipo histológico en el conjunto de prueba: {:.2f}".format((2 * results[5] * results[6]) /
                                                                                    (results[5] + results[6])))

""" -------------------------------------------------------------------------------------------------------------------
--------------------------------------------- SECCIÓN DE EVALUACIÓN  --------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Por último, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión 
de la red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos 
positivos. Además, se dibujan las curvas AUC-ROC y AUC-PR para cada uno de los datos anatomopatológicos."""
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
from scipy import interp

""" -------------------------------------------------------------------------------------------------------------------
----------------------------------------------- Tipo histológico ------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# Matriz de confusión del tipo histológico
y_true_tumor_type = []
for label_test_tumor_type in test_labels_tumor_type:
    y_true_tumor_type.append(np.argmax(label_test_tumor_type))

y_true_tumor_type = np.array(y_true_tumor_type)
y_pred_tumor_type = np.argmax(model.predict(test_image_data), axis = 1)

matrix_tumor_type = confusion_matrix(y_true_tumor_type, y_pred_tumor_type, labels = [0, 1, 2, 3, 4])
matrix_tumor_type_classes = ['IDC', 'ILC', 'Medullary', 'Metaplastic', 'Mucinous']

# Curvas ROC del tipo histológico
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
y_pred_tumor_type_prob = model.predict(test_image_data)

for i in range(len(matrix_tumor_type_classes)):
    if len(np.unique(test_labels_tumor_type[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_tumor_type[:, i], y_pred_tumor_type_prob[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_tumor_type.ravel(), y_pred_tumor_type_prob.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for tumor type')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_tumor_type_classes)):
    if len(np.unique(test_labels_tumor_type[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_tumor_type[:, i], y_pred_tumor_type_prob[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_tumor_type.ravel(),
                                                                y_pred_tumor_type_prob.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curva AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for tumor type')
plt.legend(loc = 'best')
plt.show()

""" Función para mostrar por pantalla la matriz de confusión multiclase con todas las clases de subtipos moleculares """
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión', cmap = plt.cm.Blues):
    """ Imprime y dibuja la matriz de confusión. Se puede normalizar escribiendo el parámetro `normalize=True`. """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
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

np.set_printoptions(precision=2)
fig1 = plt.figure(figsize=(7,6))

plot_confusion_matrix(matrix_tumor_type, classes = matrix_tumor_type_classes, title = 'Matriz de confusión del tipo '
                                                                                      'histológico')
plt.show()