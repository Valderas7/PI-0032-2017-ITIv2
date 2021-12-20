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

""" Se carga el modelo de red neuronal entrenado y los distintos datos de entrada y datos de salida guardados en formato 
'numpy' """
model = load_model('/mutations/image/PIK3CA SNV/inference/models/model_image_pik3ca_02_0.66_roc_f1.h5')

test_image_data = np.load('/mutations/image/PIK3CA SNV/inference/test_data/test_image.npy')
test_labels_pik3ca = np.load('/mutations/image/PIK3CA SNV/inference/test_data/test_labels_pik3ca.npy')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data, test_labels_pik3ca, verbose = 0)

""" -------------------------------------------------------------------------------------------------------------------
------------------------------------------- SECCIÓN DE EVALUACIÓN  ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
print("\n'Loss' de las mutaciones SNV del gen PIK3CA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "SNV del gen PIK3CA en el conjunto de prueba: {:.2f}%\n""Precisión de las mutaciones SNV del gen PIK3CA en el "
      "conjunto de prueba: {:.2f}%\n""Especifidad de las mutaciones SNV del gen PIK3CA en el conjunto de prueba: {:.2f}% \n"
      "Exactitud de las mutaciones SNV del gen PIK3CA en el conjunto de prueba: {:.2f}%\n""AUC-ROC de las mutaciones SNV"
      " del gen PIK3CA en el conjunto de prueba: {:.2f}\nAUC-PR de las mutaciones SNV del gen PIK3CA en el conjunto de "
      "prueba: {:.2f}".format(results[0], results[5] * 100, results[6] * 100, (results[3]/(results[3]+results[2])) * 100,
                              results[7] * 100, results[8], results[9]))

if results[5] > 0 or results[6] > 0:
    print("Valor-F de las mutaciones SNV del gen PIK3CA en el conjunto de "
          "prueba: {:.2f}".format((2 * results[5] * results[6]) / (results[5] + results[6])))

""" Para terminar, se calculan las curvas ROC. """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve

y_true = test_labels_pik3ca
y_pred_prob = model.predict(test_image_data)

# AUC-ROC
fpr, tpr, thresholds_snv_pik3ca = roc_curve(y_true, y_pred_prob)
auc_roc = auc(fpr, tpr)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for SNV of PIK3CA gene')
plt.legend(loc='best')
plt.show()

# AUC-PR
precision, recall, threshold_snv_pik3ca = precision_recall_curve(y_true, y_pred_prob)
auc_pr = auc(recall, precision)
plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for SNV of PIK3CA gene')
plt.legend(loc='best')
plt.show()