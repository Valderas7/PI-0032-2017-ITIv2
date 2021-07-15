import shelve # datos persistentes
import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
from tensorflow.keras import backend as K
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input # Para instanciar tensores de Keras
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.preprocessing import MinMaxScaler # Para escalar valores
from sklearn.metrics import confusion_matrix # Para realizar la matriz de confusión
from pytorch_grad_cam import GradCAM # Hace falta instalar pytorch, ttach, torchvision, tqdm y el propio GRAD-CAM

alto = 315 # 630
ancho = 740 # 1480
canales = 3 # Imágenes a color (RGB) = 3

model = load_model('/home/avalderas/img_slides/patient_status/distant_metastasis/inference/image/test_data&models/model_snv_image_tp53_epoch02.h5')

test_image_data = np.load('/home/avalderas/img_slides/patient_status/distant_metastasis/inference/image/test_data&models/test_image.npy')
test_labels = np.load('/home/avalderas/img_slides/patient_status/distant_metastasis/inference/image/test_data&models/test_labels.npy')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data,test_labels, verbose = 0)
print("\n'Loss' del conjunto de prueba: {:.2f}\n""Sensibilidad del conjunto de prueba: {:.2f}\n" 
      "Precisión del conjunto de prueba: {:.2f}\n""Exactitud del conjunto de prueba: {:.2f} %\n"
      "El AUC ROC del conjunto de prueba es de: {:.2f}".format(results[0], results[5], results[6], results[7] * 100,
                                                               results[8]))

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
y_true = test_labels # Etiquetas verdaderas de 'test'
y_pred = np.round(model.predict(test_image_data)) # Predicción de etiquetas de 'test'

matrix = confusion_matrix(y_true, y_pred) # Calcula (pero no dibuja) la matriz de confusión

group_names = ['True Neg','False Pos','False Neg','True Pos'] # Nombres de los grupos
group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()] # Cantidad de casos por grupo

""" @zip: Une las tuplas del nombre de los grupos con la de la cantidad de casos por grupo """
labels = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
plt.show() # Muestra la gráfica de la matriz de confusión

""" Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases.
Para implementarlas, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva: """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve

y_pred_prob = model.predict(test_image_data).ravel()
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
auc_roc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision, recall, threshold = precision_recall_curve(y_true, y_pred_prob)
auc_pr = auc(recall, precision)

plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve')
plt.legend(loc = 'best')
plt.show()