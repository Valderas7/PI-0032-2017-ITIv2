""" Librerías """
import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input # Para instanciar tensores de Keras
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.model_selection import train_test_split # Se importa la librería para dividir los datos en entreno y test.
from sklearn.preprocessing import MinMaxScaler # Para escalar valores
from sklearn.metrics import confusion_matrix # Para realizar la matriz de confusión

""" Se carga el Excel """
data_inibica = pd.read_excel('/home/avalderas/img_slides/excels/inference_inibica.xlsx', engine='openpyxl')

""" Se sustituyen los valores de la columna del estado de supervivencia, puesto que se entrenaron para valores de '1' 
para los pacientes fallecidos, al contrario que en el Excel de los pacientes de INiBICA """
data_inibica.loc[data_inibica.Estado_supervivencia == 1, "Estado_supervivencia"] = 2
data_inibica.loc[data_inibica.Estado_supervivencia == 0, "Estado_supervivencia"] = 1
data_inibica.loc[data_inibica.Estado_supervivencia == 2, "Estado_supervivencia"] = 0

""" Se crea la misma cantidad de columnas para los tipos de tumor que se creo en el conjunto de entrenamiento para 
rellenarlas posteriormente con un '1' en las filas que corresponda, dependiendo de los tipos de tumor de los pacientes 
de INiBICA. """
data_inibica['Ductal [tumor type]'] = 0
data_inibica['Lobular [tumor type]'] = 0
data_inibica['Medullary [tumor type]'] = 0
data_inibica['Metaplastic [tumor type]'] = 0
data_inibica['Mixed [tumor type]'] = 0
data_inibica['Mucinous [tumor type]'] = 0
data_inibica['Other [tumor type]'] = 0

data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Medullary'), 'Medullary [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Mucinous'), 'Mucinous [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == "Lobular"), 'Lobular [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'IDC') | (data_inibica['Tipo_tumor'] == 'Non Infiltrating Ductal'),
                 'Ductal [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Mixed'), 'Mixed [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Apocrine') | (data_inibica['Tipo_tumor'] == 'Micropapillar') |
                 (data_inibica['Tipo_tumor'] == 'Tubular'), 'Tumor_Other'] = 1

""" Se crea la misma cantidad de columnas para la variable 'STAGE' que se creo en el conjunto de entrenamiento para 
rellenarlas posteriormente con un '1' en las filas que corresponda, dependiendo de la fase de 'STAGE' de los pacientes 
de INiBICA. """
data_inibica['STAGE IB'] = 0
data_inibica['STAGE II'] = 0
data_inibica['STAGE IIA'] = 0
data_inibica['STAGE IIB'] = 0
data_inibica['STAGE III'] = 0
data_inibica['STAGE IIIA'] = 0
data_inibica['STAGE IIIB'] = 0
data_inibica['STAGE IIIC'] = 0
data_inibica['STAGE IV'] = 0
data_inibica['STAGE X'] = 0

data_inibica.loc[(data_inibica['STAGE'] == 'IB'), 'STAGE IB'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIA'), 'STAGE IIA'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIB'), 'STAGE IIB'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIIA'), 'STAGE IIIA'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIIB'), 'STAGE IIIB'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIIC'), 'STAGE IIIC'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IV'), 'STAGE IV'] = 1

""" Se binariza la columna de receptores Her-2 """
data_inibica.loc[(data_inibica['Her-2'] == '1+') | (data_inibica['Her-2'] == 0) | (data_inibica['Her-2'] == '2+')] = 0
data_inibica.loc[(data_inibica['Her-2'] == '3+')] = 1

""" Se eliminan las columnas que no sirven """
data_inibica = data_inibica.drop(['Diagnóstico_previo', 'PR', 'ER', 'pT', 'pN', 'pM', 'Ki-67', 'Tipo_tumor', 'Recidivas',
                                  'Estado_supervivencia'], axis = 1)
data_inibica = data_inibica[data_inibica.columns.drop(list(data_inibica.filter(regex='NORMAL')))]

""" Se ordenan las columnas de igual manera en el que fueron colocadas durante el proceso de entrenamiento. """
cols = data_inibica.columns.tolist()
cols = cols[:2] + cols[3:4] + cols[2:3] + cols[-18:-11] + cols[-11:]
data_inibica = data_inibica[cols]

#data_inibica.to_excel('inference_inibica_metastasis.xlsx')

""" Se carga el Excel de nuevo ya que anteriormente se ha guardado """
data_inibica_metastasis = pd.read_excel('/home/avalderas/img_slides/patient_status/data/distant_metastasis/inference/test_data&models/inference_inibica_metastasis.xlsx', engine='openpyxl')

""" Ahora habria que eliminar la columna de pacientes y guardar la columna de metastasis a distancia como variable de 
salida. """
data_inibica_metastasis = data_inibica_metastasis.drop(['Paciente'], axis = 1)
inibica_labels = data_inibica_metastasis.pop('Metástasis_distancia')

""" Se transforman ambos dataframes en formato numpy para que se les pueda aplicar la inferencia del modelo de la red 
neuronal """
test_data_inibica = np.asarray(data_inibica_metastasis).astype('float32')
inibica_labels = np.asarray(inibica_labels)

metastasis_model = load_model(
    '/clinical_data/data/distant_metastasis/inference/test_data&models/data_model_distant_metastasis_prediction.h5')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento """
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = metastasis_model.evaluate(test_data_inibica, inibica_labels, verbose = 0)
print("\n'Loss' del conjunto de prueba: {:.2f}\n""Sensibilidad del conjunto de prueba: {:.2f}\n"
      "Precisión del conjunto de prueba: {:.2f}\n""Especifidad del conjunto de prueba: {:.2f} \n"
      "Exactitud del conjunto de prueba: {:.2f} %\n"
      "El AUC-ROC del conjunto de prueba es de: {:.2f}".format(results[0], results[5], results[6],
                                                               results[3]/(results[3]+results[2]), results[7] * 100,
                                                               results[8]))

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
y_true = inibica_labels # Etiquetas verdaderas de 'test'
y_pred = np.round(metastasis_model.predict(test_data_inibica)) # Predicción de etiquetas de 'test'

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

y_pred_prob = metastasis_model.predict(test_data_inibica).ravel()
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