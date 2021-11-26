""" Programa para evaluar las métricas del subconjunto de test (conjunto total de imágenes de TCGA) de datos
anatomopatológicos.

Table 2. Anatomical pathology samples in TCGA dataset with lymph node patients (552 patients)

|       Tumor Type        |       STAGE         |    pT     |       pN      |      pM       |       IHQ     |
|   Type  |     Samples   | STAGE  |   Samples ||    Gene  | Samples |
    IDC         402         MYC         81          BRCA2       5
    ILC         105         CCND1       84          BRCA1       3
    Other       20          CDKN1B      1           KDR         1
Mixed(NOS)      17          FGF19       80          CHEK1       5
Mucinous        6          ERBB2       79          FGF3        0
Medillary       1           FGF3        81          FANCA       12
Metaplastic     1
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
model = load_model('/home/avalderas/img_slides/anatomical_pathology_data/image/inference/models/model_image_anatomical_pathology.h5')

test_image_data = np.load('/home/avalderas/img_slides/anatomical_pathology_data/image/inference/test_data/test_image.npy')

test_labels_tumor_type = np.load('/home/avalderas/img_slides/anatomical_pathology_data/image/inference/test_data/test_labels_tumor_type.npy')
test_labels_STAGE = np.load('/home/avalderas/img_slides/anatomical_pathology_data/image/inference/test_data/test_labels_STAGE.npy')
test_labels_pT = np.load('/home/avalderas/img_slides/anatomical_pathology_data/image/inference/test_data/test_labels_pT.npy')
test_labels_pN = np.load('/home/avalderas/img_slides/anatomical_pathology_data/image/inference/test_data/test_labels_pN.npy')
test_labels_pM = np.load('/home/avalderas/img_slides/anatomical_pathology_data/image/inference/test_data/test_labels_pM.npy')
test_labels_IHQ = np.load('/home/avalderas/img_slides/anatomical_pathology_data/image/inference/test_data/test_labels_IHQ.npy')

""" Una vez entrenado el modelo se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data, [test_labels_tumor_type, test_labels_STAGE, test_labels_pT, test_labels_pN,
                                           test_labels_pM, test_labels_IHQ], verbose=0)

print("\n'Loss' del tipo histológico en el conjunto de prueba: {:.2f}\n""Sensibilidad del tipo histológico en el "
      "conjunto de prueba: {:.2f}%\n""Precisión del tipo histológico en el conjunto de prueba: {:.2f}%\n""Especifidad "
      "del tipo histológico en el conjunto de prueba: {:.2f}% \n""Exactitud del tipo histológico en el conjunto de prueba: "
      "{:.2f}%\n""AUC-ROC del tipo histológico en el conjunto de prueba: {:.2f}\n""AUC-PR del tipo histológico en el "
      "conjunto de prueba: {:.2f}".format(results[1], results[11] * 100, results[12] * 100,
                                          (results[9]/(results[9]+results[8])) * 100, results[13] * 100, results[14],
                                          results[15]))
if results[11] > 0 or results[12] > 0:
    print("Valor-F del tipo histológico en el conjunto de prueba: {:.2f}".format((2 * results[11] * results[12]) /
                                                                                    (results[11] + results[12])))

print("\n'Loss' del estadio anatomopatológico en el conjunto de prueba: {:.2f}\n""Sensibilidad del estadio "
      "anatomopatológico en el conjunto de prueba: {:.2f}%\n""Precisión del estadio anatomopatológico en el conjunto de "
      "prueba: {:.2f}%\n""Especifidad del estadio anatomopatológico en el conjunto de prueba: {:.2f}% \n""Exactitud del "
      "estadio anatomopatológico en el conjunto de prueba: {:.2f}%\n""AUC-ROC del estadio anatomopatológico en el "
      "conjunto de prueba: {:.2f}\n""AUC-PR del estadio anatomopatológico en el conjunto "
      "de prueba: {:.2f}".format(results[2], results[20] * 100, results[21] * 100,
                                 (results[18]/(results[18]+results[17])) * 100, results[22] * 100, results[23],
                                 results[24]))

if results[20] > 0 or results[21] > 0:
    print("Valor-F del estadio anatomopatológico en el conjunto de prueba: {:.2f}".format((2 * results[20] * results[21]) /
                                                                                            (results[20] + results[21])))

print("\n'Loss' del parámetro 'T' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'T' en el conjunto de "
      "prueba: {:.2f}%\n""Precisión del parámetro 'T' en el conjunto de prueba: {:.2f}%\n""Especifidad del parámetro 'T' "
      "en el conjunto de prueba: {:.2f}% \n""Exactitud del parámetro 'T' en el conjunto de prueba: {:.2f}%\n""AUC-ROC "
      "del parámetro 'T' en el conjunto de prueba: {:.2f}\n""AUC-PR del parámetro 'T' en el conjunto de "
      "prueba: {:.2f}".format(results[3], results[29] * 100, results[30] * 100,
                              (results[27]/(results[27]+results[26])) * 100, results[31] * 100, results[32],
                              results[33]))

if results[29] > 0 or results[30] > 0:
    print("Valor-F del parámetro 'T' en el conjunto de prueba: {:.2f}".format((2 * results[29] * results[30]) /
                                                                                (results[29] + results[30])))

print("\n'Loss' del parámetro 'N' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'N' en el conjunto de "
      "prueba: {:.2f}%\n""Precisión del parámetro 'N' en el conjunto de prueba: {:.2f}%\n""Especifidad del parámetro 'N' "
      "en el conjunto de prueba: {:.2f}%\n""Exactitud del parámetro 'N' en el conjunto de prueba: {:.2f}%\n""AUC-ROC "
      "del parámetro 'N' en el conjunto de prueba: {:.2f}\n""AUC-PR del parámetro 'N' en el conjunto de "
      "prueba: {:.2f}".format(results[4], results[38] * 100, results[39] * 100,
                              (results[36]/(results[36]+results[35])) * 100, results[40] * 100, results[41],
                              results[42]))

if results[38] > 0 or results[39] > 0:
    print("Valor-F del parámetro 'N' en el conjunto de prueba: {:.2f}".format((2 * results[38] * results[39]) /
                                                                                (results[38] + results[39])))

print("\n'Loss' del parámetro 'M' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'M' en el conjunto de "
      "prueba: {:.2f}%\n""Precisión del parámetro 'M' en el conjunto de prueba: {:.2f}%\n""Especifidad del parámetro 'M' "
      "en el conjunto de prueba: {:.2f}% \n""Exactitud del parámetro 'M' en el conjunto de prueba: {:.2f}%\n""AUC-ROC "
      "del parámetro 'M' en el conjunto de prueba: {:.2f}\n""AUC-PR del parámetro 'M' en el conjunto de "
      "prueba: {:.2f}".format(results[5], results[47] * 100, results[48] * 100,
                              (results[45]/(results[45]+results[44])) * 100, results[49] * 100, results[50],
                              results[51]))

if results[47] > 0 or results[48] > 0:
    print("Valor-F del parámetro 'M' en el conjunto de prueba: {:.2f}".format((2 * results[47] * results[48]) /
                                                                                (results[47] + results[48])))

print("\n'Loss' del subtipo molecular en el conjunto de prueba: {:.2f}\n""Sensibilidad del subtipo molecular en el "
      "conjunto de prueba: {:.2f}%\n""Precisión del subtipo molecular en el conjunto de prueba: {:.2f}%\n""Especifidad "
      "del subtipo molecular en el conjunto de prueba: {:.2f}%\n""Exactitud del subtipo molecular en el conjunto de "
      "prueba: {:.2f}%\n""AUC-ROC del subtipo molecular en el conjunto de prueba: {:.2f}\n""AUC-PR del subtipo molecular "
      "en el conjunto de prueba: {:.2f}".format(results[6], results[56] * 100, results[57] * 100,
                                                (results[54]/(results[54]+results[53])) * 100, results[58] * 100,
                                                results[59], results[60]))

if results[56] > 0 or results[57] > 0:
    print("Valor-F del subtipo molecular en el conjunto de prueba: {:.2f}".format((2 * results[56] * results[57]) /
                                                                                    (results[56] + results[57])))

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
y_pred_tumor_type = np.argmax(model.predict(test_image_data)[0], axis = 1)

matrix_tumor_type = confusion_matrix(y_true_tumor_type, y_pred_tumor_type, labels = [0, 1, 2, 3, 4, 5, 6])
matrix_tumor_type_classes = ['IDC', 'ILC', 'Medullary', 'Metaplastic', 'Mixed (NOS)', 'Mucinous', 'Other']

# Curvas ROC del tipo histológico
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
y_pred_tumor_type_prob = model.predict(test_image_data)[0]

for i in range(len(matrix_tumor_type_classes)):
    if len(np.unique(test_labels_tumor_type[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_tumor_type[:, i], y_pred_tumor_type_prob[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_tumor_type.ravel(), y_pred_tumor_type_prob.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
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
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for tumor type')
plt.legend(loc = 'best')
plt.show()

""" -------------------------------------------------------------------------------------------------------------------
------------------------------------------- Estadio anatomopatológico -------------------------------------------------
-------------------------------------------------------------------------------------------------------------------- """
# Matriz de confusión del estadio anatomopatológico
y_true_STAGE = []
for label_test_STAGE in test_labels_STAGE:
    y_true_STAGE.append(np.argmax(label_test_STAGE))

y_true_STAGE = np.array(y_true_STAGE)
y_pred_STAGE = np.argmax(model.predict(test_image_data)[1], axis = 1)

matrix_STAGE = confusion_matrix(y_true_STAGE, y_pred_STAGE, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Calcula (pero no dibuja) la matriz de confusión
matrix_STAGE_classes = ['Stage IB', 'Stage II', 'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IIIA', 'Stage IIIB',
                        'Stage IIIC', 'Stage IV', 'STAGE X']

# Curvas ROC del estadio anatomopatológico
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
y_pred_STAGE_prob = model.predict(test_image_data)[1]

for i in range(len(matrix_STAGE_classes)):
    if len(np.unique(test_labels_STAGE[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_STAGE[:, i], y_pred_STAGE_prob[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_STAGE.ravel(), y_pred_STAGE_prob.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for STAGE')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_STAGE_classes)):
    if len(np.unique(test_labels_STAGE[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_STAGE[:, i], y_pred_STAGE_prob[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_STAGE.ravel(), y_pred_STAGE_prob.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curva AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for STAGE')
plt.legend(loc = 'best')
plt.show()

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------------------- Parámetro pT -----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# Matriz de confusión del parámetro pT
y_true_pT = []
for label_test_pT in test_labels_pT:
    y_true_pT.append(np.argmax(label_test_pT))

y_true_pT = np.array(y_true_pT)
y_pred_pT = np.argmax(model.predict(test_image_data)[2], axis = 1)

matrix_pT = confusion_matrix(y_true_pT, y_pred_pT, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pT_classes = ['T1', 'T1A', 'T1B', 'T1C', 'T2', 'T2B', 'T3', 'T4', 'T4B', 'T4D']

# Curvas ROC del parámetro pT
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
y_pred_pT_prob = model.predict(test_image_data)[2]

for i in range(len(matrix_pT_classes)):
    if len(np.unique(test_labels_pT[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_pT[:, i], y_pred_pT_prob[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_pT.ravel(), y_pred_pT_prob.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for pT parameter')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_pT_classes)):
    if len(np.unique(test_labels_pT[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_pT[:, i], y_pred_pT_prob[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_pT.ravel(), y_pred_pT_prob.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curva AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for pT parameter')
plt.legend(loc = 'best')
plt.show()

""" -------------------------------------------------------------------------------------------------------------------
-------------------------------------------------- Parámetro pN -------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# Matriz de confusion del parametro pN
y_true_pN = []
for label_test_pN in test_labels_pN:
    y_true_pN.append(np.argmax(label_test_pN))

y_true_pN = np.array(y_true_pN)
y_pred_pN = np.argmax(model.predict(test_image_data)[3], axis = 1)

matrix_pN = confusion_matrix(y_true_pN, y_pred_pN, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pN_classes = ['N1', 'N1A', 'N1B', 'N1C', 'N1MI', 'N2', 'N2A', 'N3', 'N3A', 'N3B', 'N3C']

# Curvas ROC del parametro pN
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
y_pred_pN_prob = model.predict(test_image_data)[3]

for i in range(len(matrix_pN_classes)):
    if len(np.unique(test_labels_pN[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_pN[:, i], y_pred_pN_prob[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_pN.ravel(), y_pred_pN_prob.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for pN parameter')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_pN_classes)):
    if len(np.unique(test_labels_pN[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_pN[:, i], y_pred_pN_prob[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_pN.ravel(), y_pred_pN_prob.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curva AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for pN parameter')
plt.legend(loc = 'best')
plt.show()

""" -------------------------------------------------------------------------------------------------------------------
--------------------------------------------------- Parámetro pM ------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# Matriz de confusion del parametro pM
y_true_pM = []
for label_test_pM in test_labels_pM:
    y_true_pM.append(np.argmax(label_test_pM))

y_true_pM = np.array(y_true_pM)
y_pred_pM = np.argmax(model.predict(test_image_data)[4], axis = 1)

matrix_pM = confusion_matrix(y_true_pM, y_pred_pM, labels = [0, 1, 2])
matrix_pM_classes = ['M0', 'M1', 'MX']

# Curvas ROC del parametro pM
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
y_pred_pM_prob = model.predict(test_image_data)[4]

for i in range(len(matrix_pM_classes)):
    if len(np.unique(test_labels_pM[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_pM[:, i], y_pred_pM_prob[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_pM.ravel(), y_pred_pM_prob.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for pM parameter')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_pM_classes)):
    if len(np.unique(test_labels_pM[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_pM[:, i], y_pred_pM_prob[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_pM.ravel(), y_pred_pM_prob.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curva AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for pM parameter')
plt.legend(loc = 'best')
plt.show()

""" -------------------------------------------------------------------------------------------------------------------
----------------------------------------------- Inmunohistoquímica ----------------------------------------------------
-------------------------------------------------------------------------------------------------------------------- """
# Matriz de confusion de la IHQ
y_true_IHQ = []
for label_test_IHQ in test_labels_IHQ:
    y_true_IHQ.append(np.argmax(label_test_IHQ))

y_true_IHQ = np.array(y_true_IHQ)
y_pred_IHQ = np.argmax(model.predict(test_image_data)[5], axis = 1)

matrix_IHQ = confusion_matrix(y_true_IHQ, y_pred_IHQ, labels = [0, 1, 2, 3, 4]) # Calcula (pero no dibuja) la matriz de confusión
matrix_IHQ_classes = ['Basal', 'Her2', 'Luminal A', 'Luminal B', 'Normal']

# Curvas ROC del tipo histológico
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
y_pred_IHQ_prob = model.predict(test_image_data)[5]

for i in range(len(matrix_IHQ_classes)):
    if len(np.unique(test_labels_IHQ[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_IHQ[:, i], y_pred_IHQ_prob[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_IHQ.ravel(), y_pred_IHQ_prob.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for IHQ')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_IHQ_classes)):
    if len(np.unique(test_labels_IHQ[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_IHQ[:, i], y_pred_IHQ_prob[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_IHQ.ravel(), y_pred_IHQ_prob.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curva AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for IHQ')
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

plot_confusion_matrix(matrix_STAGE, classes = matrix_STAGE_classes, title = 'Matriz de confusión del estadio '
                                                                            'anatomopatológico')
plt.show()

plot_confusion_matrix(matrix_pT, classes = matrix_pT_classes, title = 'Matriz de confusión del parámetro "T"')
plt.show()

plot_confusion_matrix(matrix_pN, classes = matrix_pN_classes, title = 'Matriz de confusión del parámetro "N"')
plt.show()

plot_confusion_matrix(matrix_pM, classes = matrix_pM_classes, title = 'Matriz de confusión del parámetro "M"')
plt.show()

plot_confusion_matrix(matrix_IHQ, classes = matrix_IHQ_classes, title = 'Matriz de confusión del subtipo molecular')
plt.show()