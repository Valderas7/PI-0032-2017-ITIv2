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

""" Se carga el modelo de red neuronal entrenado y los distintos datos de entrada y datos de salida guardados en formato 
'numpy' """
model = load_model('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/mutations-anatomopathologic.h5')

test_tabular_data = np.load('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/test_data.npy')

test_labels_tumor_type = np.load('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/test_labels_tumor_type.npy')
test_labels_STAGE = np.load('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/test_labels_STAGE.npy')
test_labels_pT = np.load('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/test_labels_pT.npy')
test_labels_pN = np.load('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/test_labels_pN.npy')
test_labels_pM = np.load('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/test_labels_pM.npy')
test_labels_IHQ = np.load('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/test_labels_IHQ.npy')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_tabular_data, [test_labels_tumor_type, test_labels_STAGE, test_labels_pT, test_labels_pN,
                                             test_labels_pM, test_labels_IHQ], verbose = 0)

print("\n'Loss' del tipo histológico en el conjunto de prueba: {:.2f}\n""Sensibilidad del tipo histológico en el "
      "conjunto de prueba: {:.2f}\n""Precisión del tipo histológico en el conjunto de prueba: {:.2f}\n""Especifidad del "
      "tipo histológico en el conjunto de prueba: {:.2f} \n""Exactitud del tipo histológico en el conjunto de prueba: "
      "{:.2f} %\n""AUC-ROC del tipo histológico en el conjunto de prueba: {:.2f}".format(results[1], results[11],
                                                                                           results[12],
                                                                                           results[9]/(results[9]+results[8]),
                                                                                           results[13] * 100, results[14]))
if results[11] > 0 or results[12] > 0:
    print("Valor-F del tipo histológico en el conjunto de prueba: {:.2f}".format((2 * results[11] * results[12]) /
                                                                                    (results[11] + results[12])))

print("\n'Loss' del estadio anatomopatológico en el conjunto de prueba: {:.2f}\n""Sensibilidad del estadio "
      "anatomopatológico en el conjunto de prueba: {:.2f}\n""Precisión del estadio anatomopatológico en el conjunto de "
      "prueba: {:.2f}\n""Especifidad del estadio anatomopatológico en el conjunto de prueba: {:.2f} \n""Exactitud del "
      "estadio anatomopatológico en el conjunto de prueba: {:.2f} %\n""AUC-ROC del estadio anatomopatológico en el "
      "conjunto de prueba: {:.2f}".format(results[2], results[19], results[20], results[17]/(results[17]+results[16]),
                                          results[21] * 100, results[22]))
if results[19] > 0 or results[20] > 0:
    print("Valor-F del estadio anatomopatológico en el conjunto de prueba: {:.2f}".format((2 * results[19] * results[20]) /
                                                                                            (results[19] + results[20])))

print("\n'Loss' del parámetro 'T' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'T' en el conjunto de "
      "prueba: {:.2f}\n""Precisión del parámetro 'T' en el conjunto de prueba: {:.2f}\n""Especifidad del parámetro 'T' "
      "en el conjunto de prueba: {:.2f} \n""Exactitud del parámetro 'T' en el conjunto de prueba: {:.2f} %\n""AUC-ROC "
      "del parámetro 'T' en el conjunto de prueba: {:.2f}".format(results[3], results[27], results[28],
                                                                  results[25]/(results[25]+results[24]),
                                                                  results[29] * 100, results[30]))
if results[27] > 0 or results[28] > 0:
    print("Valor-F del parámetro 'T' en el conjunto de prueba: {:.2f}".format((2 * results[27] * results[28]) /
                                                                                (results[27] + results[28])))

print("\n'Loss' del parámetro 'N' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'N' en el conjunto de "
      "prueba: {:.2f}\n""Precisión del parámetro 'N' en el conjunto de prueba: {:.2f}\n""Especifidad del parámetro 'N' "
      "en el conjunto de prueba: {:.2f} \n""Exactitud del parámetro 'N' en el conjunto de prueba: {:.2f} %\n""AUC-ROC "
      "del parámetro 'N' en el conjunto de prueba: {:.2f}".format(results[4], results[35], results[36],
                                                                  results[33]/(results[33]+results[32]),
                                                                  results[37] * 100, results[38]))
if results[35] > 0 or results[36] > 0:
    print("Valor-F del parámetro 'N' en el conjunto de prueba: {:.2f}".format((2 * results[35] * results[36]) /
                                                                                (results[35] + results[36])))

print("\n'Loss' del parámetro 'M' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'M' en el conjunto de "
      "prueba: {:.2f}\n""Precisión del parámetro 'M' en el conjunto de prueba: {:.2f}\n""Especifidad del parámetro 'M' "
      "en el conjunto de prueba: {:.2f} \n""Exactitud del parámetro 'M' en el conjunto de prueba: {:.2f} %\n""AUC-ROC "
      "del parámetro 'M' en el conjunto de prueba: {:.2f}".format(results[5], results[43], results[44],
                                                                  results[41]/(results[41]+results[40]),
                                                                  results[45] * 100, results[46]))
if results[43] > 0 or results[44] > 0:
    print("Valor-F del parámetro 'M' en el conjunto de prueba: {:.2f}".format((2 * results[43] * results[44]) /
                                                                                (results[43] + results[44])))

print("\n'Loss' del subtipo molecular en el conjunto de prueba: {:.2f}\n""Sensibilidad del subtipo molecular en el "
      "conjunto de prueba: {:.2f}\n""Precisión del subtipo molecular en el conjunto de prueba: {:.2f}\n""Especifidad del "
      "subtipo molecular en el conjunto de prueba: {:.2f}\n""Exactitud del subtipo molecular en el conjunto de prueba: "
      "{:.2f} %\n""AUC-ROC del subtipo molecular en el conjunto de prueba: {:.2f}".format(results[6], results[51],
                                                                                          results[52],
                                                                                          results[49]/(results[49]+results[48]),
                                                                                          results[53] * 100,
                                                                                          results[54]))
if results[51] > 0 or results[52] > 0:
    print("Valor-F del subtipo molecular en el conjunto de prueba: {:.2f}".format((2 * results[51] * results[52]) /
                                                                                    (results[51] + results[52])))

""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos.
Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
# Tipo histológico
y_true_tumor_type = []
for label_test_tumor_type in test_labels_tumor_type:
    y_true_tumor_type.append(np.argmax(label_test_tumor_type))

y_true_tumor_type = np.array(y_true_tumor_type)
y_pred_tumor_type = np.argmax(model.predict(test_tabular_data)[0], axis = 1)

matrix_tumor_type = confusion_matrix(y_true_tumor_type, y_pred_tumor_type, labels = [0, 1, 2, 3, 4, 5, 6])
matrix_tumor_type_classes = ['IDC', 'ILC', 'Medullary', 'Metaplastic', 'Mixed (NOS)', 'Mucinous', 'Other']

# Estadio anatomopatológico
y_true_STAGE = []
for label_test_STAGE in test_labels_STAGE:
    y_true_STAGE.append(np.argmax(label_test_STAGE))

y_true_STAGE = np.array(y_true_STAGE)
y_pred_STAGE = np.argmax(model.predict(test_tabular_data)[1], axis = 1)

matrix_STAGE = confusion_matrix(y_true_STAGE, y_pred_STAGE, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Calcula (pero no dibuja) la matriz de confusión
matrix_STAGE_classes = ['Stage IB', 'Stage II', 'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IIIA', 'Stage IIIB',
                        'Stage IIIC', 'Stage IV', 'STAGE X']

# pT
y_true_pT = []
for label_test_pT in test_labels_pT:
    y_true_pT.append(np.argmax(label_test_pT))

y_true_pT = np.array(y_true_pT)
y_pred_pT = np.argmax(model.predict(test_tabular_data)[2], axis = 1)

matrix_pT = confusion_matrix(y_true_pT, y_pred_pT, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pT_classes = ['T1', 'T1A', 'T1B', 'T1C', 'T2', 'T2B', 'T3', 'T4', 'T4B', 'T4D']

# pN
y_true_pN = []
for label_test_pN in test_labels_pN:
    y_true_pN.append(np.argmax(label_test_pN))

y_true_pN = np.array(y_true_pN)
y_pred_pN = np.argmax(model.predict(test_tabular_data)[3], axis = 1)

matrix_pN = confusion_matrix(y_true_pN, y_pred_pN, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pN_classes = ['N1', 'N1A', 'N1B', 'N1C', 'N1MI', 'N2', 'N2A', 'N3', 'N3A', 'N3B', 'N3C']

# pM
y_true_pM = []
for label_test_pM in test_labels_pM:
    y_true_pM.append(np.argmax(label_test_pM))

y_true_pM = np.array(y_true_pM)
y_pred_pM = np.argmax(model.predict(test_tabular_data)[4], axis = 1)

matrix_pM = confusion_matrix(y_true_pM, y_pred_pM, labels = [0, 1, 2])
matrix_pM_classes = ['M0', 'M1', 'MX']

# IHQ
y_true_IHQ = []
for label_test_IHQ in test_labels_IHQ:
    y_true_IHQ.append(np.argmax(label_test_IHQ))

y_true_IHQ = np.array(y_true_IHQ)
y_pred_IHQ = np.argmax(model.predict(test_tabular_data)[5], axis = 1)

matrix_IHQ = confusion_matrix(y_true_IHQ, y_pred_IHQ, labels = [0, 1, 2, 3, 4]) # Calcula (pero no dibuja) la matriz de confusión
matrix_IHQ_classes = ['Basal', 'Her2', 'Luminal A', 'Luminal B', 'Normal']

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
