""" Librerías """
import itertools
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
data_inibica.loc[data_inibica.Supervivencia == 1, "Supervivencia"] = 2
data_inibica.loc[data_inibica.Supervivencia == 0, "Supervivencia"] = 1
data_inibica.loc[data_inibica.Supervivencia == 2, "Supervivencia"] = 0

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

data_inibica.loc[(data_inibica['Tipo histológico'] == 'Medullary'), 'Medullary [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo histológico'] == 'Mucinous'), 'Mucinous [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo histológico'] == "Lobular"), 'Lobular [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo histológico'] == 'IDC') |
                 (data_inibica['Tipo histológico'] == 'Non Infiltrating Ductal'), 'Ductal [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo histológico'] == 'Mixed'), 'Mixed [tumor type]'] = 1
data_inibica.loc[(data_inibica['Tipo histológico'] == 'Apocrine')
                 | (data_inibica['Tipo histológico'] == 'Micropapillar')
                 | (data_inibica['Tipo histológico'] == 'Tubular'), 'Other [tumor type]'] = 1

""" Se crea la misma cantidad de columnas para la variable 'STAGE' que se creo en el conjunto de entrenamiento para 
rellenarlas posteriormente con un '1' en las filas que corresponda, dependiendo de la fase de 'STAGE' de los pacientes 
de INiBICA. """
data_inibica['STAGE I'] = 0
data_inibica['STAGE IA'] = 0
data_inibica['STAGE IB'] = 0
data_inibica['STAGE II'] = 0
data_inibica['STAGE IIA'] = 0
data_inibica['STAGE IIB'] = 0
data_inibica['STAGE III'] = 0
data_inibica['STAGE IIIA'] = 0
data_inibica['STAGE IIIB'] = 0
data_inibica['STAGE IIIC'] = 0
data_inibica['STAGE X'] = 0

data_inibica.loc[(data_inibica['STAGE'] == 'I'), 'STAGE I'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IA'), 'STAGE IA'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IB'), 'STAGE IB'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIA'), 'STAGE IIA'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIB'), 'STAGE IIB'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIIA'), 'STAGE IIIA'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIIB'), 'STAGE IIIB'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIIC'), 'STAGE IIIC'] = 1

""" Se binariza la columna de receptores Her-2 """
data_inibica.loc[(data_inibica['Her-2'] == '1+') | (data_inibica['Her-2'] == '0')
                 | (data_inibica['Her-2'] == '2+'), 'Her-2'] = 0
data_inibica.loc[(data_inibica['Her-2'] == '3+'), 'Her-2'] = 1

""" Se eliminan las columnas que no sirven """
data_inibica = data_inibica.drop(['Diagnóstico previo', 'PR', 'ER', 'pT', 'pN', 'pM', 'Ki-67', 'STAGE',
                                  'Tipo histológico'], axis = 1)
data_inibica = data_inibica[data_inibica.columns.drop(list(data_inibica.filter(regex='NORMAL')))]

""" Se ordenan las columnas de igual manera en el que fueron colocadas durante el proceso de entrenamiento. """
cols = data_inibica.columns.tolist()
cols = cols[:2] + cols[5:6] + cols[4:5] + cols[2:3] + cols[3:4] + cols[-18:-11] + cols[-11:] + cols[6:-18]
data_inibica = data_inibica[cols]
#data_inibica.to_excel('inference_inibica_mutations.xlsx')

""" Se carga el Excel de nuevo ya que anteriormente se ha guardado """
data_inibica_saved = pd.read_excel('/home/avalderas/img_slides/mutations/data/excels/inference_inibica_mutations.xlsx', engine='openpyxl')

""" Ahora habria que eliminar la columna de pacientes y extraer la columna de la variable de salida. """
data_inibica_saved = data_inibica_saved.drop(['Paciente'], axis = 1)
test_labels = data_inibica_saved.pop('CNV-D FANCA')

""" Se transforman ambos dataframes en formato numpy para que se les pueda aplicar la inferencia del modelo de la red 
neuronal """
test_data = np.asarray(data_inibica_saved).astype('float32')
test_labels = np.asarray(test_labels)

model = load_model('/home/avalderas/img_slides/mutations/data/CNV-D/FANCA/inference/models/model_data_FANCA_1000_0.63.h5')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento """
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_data, test_labels, verbose = 0)

print("\n'Loss' de mutación CNV-D de FANCA: {:.2f}\n""Sensibilidad de mutación CNV-D de FANCA: {:.2f}%\n""Precisión de "
      "mutación CNV-D de FANCA: {:.2f}%\n""Especificidad de mutación CNV-D de FANCA: {:.2f}% \n""Exactitud de mutación "
      "CNV-D de FANCA: {:.2f}%\n""AUC-ROC de mutación CNV-D de FANCA: {:.2f}\n"
      "AUC-PR de mutación CNV-D de FANCA: {:.2f}".format(results[0], results[5] * 100, results[6] * 100,
                                                         (results[3]/(results[3]+results[2])) * 100, results[7] * 100,
                                                         results[8], results[9]))

if results[5] > 0 or results[6] > 0:
    print("Valor-F de mutación CNV-D de FANCA: {:.2f}".format((2 * results[5] * results[6]) / (results[5] + results[6])))

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
    plt.xlabel('Predicción')

# Matriz de confusión
true_labels = test_labels
binary_predictions = np.round(model.predict(test_data))

matrix_mutation = confusion_matrix(true_labels, binary_predictions, labels = [0, 1])
matrix_mutation_classes = ['Sin mutación', 'Con mutación']

plot_confusion_matrix(matrix_mutation, classes = matrix_mutation_classes, title ='Matriz de confusión [CNV-D FANCA]')
plt.show()

""" Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases.
Para implementarlas, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva: """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve

raw_predictions = model.predict(test_data).ravel()
fpr, tpr, threshold_roc = roc_curve(test_labels, raw_predictions)
auc_roc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for CNV-D mutation of FANCA gene')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision, recall, threshold_pr = precision_recall_curve(test_labels, raw_predictions)
auc_pr = auc(recall, precision)

plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for CNV-D mutation of FANCA gene')
plt.legend(loc = 'best')
plt.show()