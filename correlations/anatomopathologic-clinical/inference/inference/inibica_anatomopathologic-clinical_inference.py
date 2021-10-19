""" Librerías """
import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input # Para instanciar tensores de Keras
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.metrics import confusion_matrix
import itertools

""" Se carga el Excel de los pacientes de INiBICA """
data_inibica = pd.read_excel('/home/avalderas/img_slides/excel_genesOCA&inibica_patients/inference_inibica.xlsx',
                             engine='openpyxl')

""" Se sustituyen los valores de la columna del estado de supervivencia, puesto que se entrenaron para valores de '1' 
para los pacientes fallecidos, al contrario que en el Excel de los pacientes de INiBICA """
data_inibica.loc[data_inibica.Estado_supervivencia == 1, "Estado_supervivencia"] = 2
data_inibica.loc[data_inibica.Estado_supervivencia == 0, "Estado_supervivencia"] = 1
data_inibica.loc[data_inibica.Estado_supervivencia == 2, "Estado_supervivencia"] = 0

""" Se crea la misma cantidad de columnas para los tipos de tumor que se creo en el conjunto de entrenamiento para 
rellenarlas posteriormente con un '1' en las filas que corresponda, dependiendo de los tipos de tumor de los pacientes 
de INiBICA. """
data_inibica['Tumor_IDC'] = 0
data_inibica['Tumor_ILC'] = 0
data_inibica['Tumor_Metaplastic'] = 0
data_inibica['Tumor_Mixed'] = 0
data_inibica['Tumor_Mucinous'] = 0
data_inibica['Tumor_Other'] = 0

data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Mucinous'), 'Tumor_Mucinous'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'].str.contains("Lobular")) |
                 (data_inibica['Tipo_tumor'] == 'Signet-ring cells lobular'), 'Tumor_ILC'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Invasive carcinoma (NST)') |
                 (data_inibica['Tipo_tumor'] == 'Microinvasive carcinoma') |
                 (data_inibica['Tipo_tumor'] == 'Signet-ring cells ductal'), 'Tumor_IDC'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Mixed ductal and lobular'), 'Tumor_Mixed'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Apocrine') | (data_inibica['Tipo_tumor'] == 'Papillary') |
                 (data_inibica['Tipo_tumor'] == 'Tubular') | (data_inibica['Tipo_tumor'] == 'Medullary'),
                 'Tumor_Other'] = 1

""" Se elimina la columna 'Tipo_tumor', ya que ya no nos sirve, al igual que las demas columnas que no se utilizan para 
la prediccion de metastasis. """
data_inibica = data_inibica.drop(['Tipo_tumor'], axis = 1)

""" Se crea la misma cantidad de columnas para la variable 'STAGE' que se creo en el conjunto de entrenamiento para 
rellenarlas posteriormente con un '1' en las filas que corresponda, dependiendo de la fase de 'STAGE' de los pacientes 
de INiBICA. """
data_inibica['STAGE_IB'] = 0
data_inibica['STAGE_II'] = 0
data_inibica['STAGE_IIA'] = 0
data_inibica['STAGE_IIB'] = 0
data_inibica['STAGE_III'] = 0
data_inibica['STAGE_IIIA'] = 0
data_inibica['STAGE_IIIB'] = 0
data_inibica['STAGE_IIIC'] = 0
data_inibica['STAGE_X'] = 0

data_inibica.loc[(data_inibica['STAGE'] == 'IB'), 'STAGE_IB'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIA'), 'STAGE_IIA'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIB'), 'STAGE_IIB'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIIA'), 'STAGE_IIIA'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIIB'), 'STAGE_IIIB'] = 1
data_inibica.loc[(data_inibica['STAGE'] == 'IIIC'), 'STAGE_IIIC'] = 1

""" Se elimina la columna 'STAGE', ya que ya no nos sirve, al igual que las demas columnas que no se utilizan para 
la prediccion de metastasis. """
data_inibica = data_inibica.drop(['STAGE'], axis = 1)

""" Se crean la misma cantidad de columnas para los estadios T que se crearon en el conjunto de entrenamiento para 
rellenarlas posteriormente con un '1' en las filas que corresponda, dependiendo del valor pT de los pacientes de INiBICA. """
data_inibica['pT_T1'] = 0
data_inibica['pT_T1B'] = 0
data_inibica['pT_T1C'] = 0
data_inibica['pT_T2'] = 0
data_inibica['pT_T2B'] = 0
data_inibica['pT_T3'] = 0
data_inibica['pT_T4'] = 0
data_inibica['pT_T4B'] = 0
data_inibica['pT_T4D'] = 0

data_inibica.loc[data_inibica.pT == 1, 'pT_T1'] = 1
data_inibica.loc[data_inibica.pT == '1b', 'pT_T1B'] = 1
data_inibica.loc[data_inibica.pT == '1c', 'pT_T1C'] = 1
data_inibica.loc[data_inibica.pT == 2, 'pT_T2'] = 1
data_inibica.loc[data_inibica.pT == 3, 'pT_T3'] = 1
data_inibica.loc[data_inibica.pT == 4, 'pT_T4'] = 1
data_inibica.loc[data_inibica.pT == '4a', 'pT_T4'] = 1
data_inibica.loc[data_inibica.pT == '4b', 'pT_T4B'] = 1
data_inibica.loc[data_inibica.pT == '4d', 'pT_T4D'] = 1

""" Se elimina la columna 'pT', ya que ya no nos sirve """
data_inibica = data_inibica.drop(['pT'], axis = 1)

""" Se crean la misma cantidad de columnas para los estadios N que se crearon en el conjunto de entrenamiento para 
rellenarlas posteriormente con un '1' en las filas que corresponda, dependiendo del valor pN de los pacientes de INiBICA. """
data_inibica['pN_N1'] = 0
data_inibica['pN_N1A'] = 0
data_inibica['pN_N1B'] = 0
data_inibica['pN_N1C'] = 0
data_inibica['pN_N1MI'] = 0
data_inibica['pN_N2'] = 0
data_inibica['pN_N2A'] = 0
data_inibica['pN_N3'] = 0
data_inibica['pN_N3A'] = 0
data_inibica['pN_N3B'] = 0

data_inibica.loc[data_inibica.pN == 1, 'pN_N1'] = 1
data_inibica.loc[data_inibica.pN == '1a', 'pN_N1A'] = 1
data_inibica.loc[data_inibica.pN == '1b', 'pN_N1B'] = 1
data_inibica.loc[data_inibica.pN == '1c', 'pN_N1C'] = 1
data_inibica.loc[data_inibica.pN == '1mi', 'pN_N1MI'] = 1
data_inibica.loc[data_inibica.pN == 2, 'pN_N2'] = 1
data_inibica.loc[data_inibica.pN == '2a', 'pN_N2A'] = 1
data_inibica.loc[data_inibica.pN == 3, 'pN_N3'] = 1
data_inibica.loc[data_inibica.pN == '3a', 'pN_N3A'] = 1

""" Se elimina la columna 'pN', ya que ya no nos sirve """
data_inibica = data_inibica.drop(['pN'], axis = 1)

""" Se crean la misma cantidad de columnas para los estadios M que se crearon en el conjunto de entrenamiento para 
rellenarlas posteriormente con un '1' en las filas que corresponda, dependiendo del valor pM de los pacientes de INiBICA. """
data_inibica['pM_M0'] = 0
data_inibica['pM_M1'] = 0
data_inibica['pM_MX'] = 0

data_inibica.loc[(data_inibica['pM'] == 0), 'pM_M0'] = 1
data_inibica.loc[(data_inibica['pM'] == 'X'), 'pM_MX'] = 1

""" Se elimina la columna 'pM', ya que ya no nos sirve """
data_inibica = data_inibica.drop(['pM'], axis = 1)

""" Se crean la misma cantidad de columnas para la IHQ que se crearon en el conjunto de entrenamiento para rellenarlas 
posteriormente con un '1' en las filas que corresponda, dependiendo del valor de los distintos receptores de los 
pacientes de INiBICA. """
data_inibica['IHQ_Basal'] = 0
data_inibica['IHQ_Her2'] = 0
data_inibica['IHQ_Luminal_A'] = 0
data_inibica['IHQ_Luminal_B'] = 0
data_inibica['IHQ_Normal'] = 0

""" Se aprecian valores nulos en la columna de 'Ki-67'. Para no desechar estos pacientes, se ha optado por considerarlos 
como pacientes con porcentaje mínimo de Ki-67: """
data_inibica['Ki-67'].fillna(value = 0, inplace= True)

""" Para el criterio de IHQ, se busca en internet el criterio para clasificar los distintos cáncer de mama en Luminal A, 
Luminal B, Her2, Basal, etc. """
data_inibica.loc[((data_inibica['ER'] > 0.01) | (data_inibica['PR'] > 0.01)) & ((data_inibica['Her-2'] == 0) |
                                                                                (data_inibica['Her-2'] == '1+')) &
                 (data_inibica['Ki-67'] < 0.14),'IHQ_Luminal_A'] = 1

data_inibica.loc[(((data_inibica['ER'] > 0.01) | (data_inibica['PR'] > 0.01)) &
                  ((data_inibica['Her-2'] == 0) | (data_inibica['Her-2'] == '1+')) & (data_inibica['Ki-67'] >= 0.14)) |
                 (((data_inibica['ER'] > 0.01) | (data_inibica['PR'] > 0.01)) &
                  ((data_inibica['Her-2'] == '2+') | (data_inibica['Her-2'] == '3+'))),'IHQ_Luminal_B'] = 1

data_inibica.loc[(data_inibica['ER'] <= 0.01) & (data_inibica['PR'] <= 0.01) &
                 ((data_inibica['Her-2'] == '2+') | (data_inibica['Her-2'] == '3+')),'IHQ_Her2'] = 1

data_inibica.loc[(data_inibica['ER'] <= 0.01) & (data_inibica['PR'] <= 0.01) &
                 ((data_inibica['Her-2'] == 0) | (data_inibica['Her-2'] == '1+')),'IHQ_Basal'] = 1

data_inibica.loc[(data_inibica['IHQ_Luminal_A'] == '0') & (data_inibica['IHQ_Luminal_B'] == '0') &
                 (data_inibica['IHQ_Her2'] == '0') & (data_inibica['IHQ_Basal'] == '0'),'IHQ_Normal'] = 1

""" Se eliminan las columnas de IHQ, ya que no sirven, y las columnas clínicas que no han sido usadas """
data_inibica = data_inibica.drop(['ER', 'PR', 'Ki-67', 'Her-2', 'Diagnóstico_previo', 'Edad',
                                  'Tratamiento_neoadyuvante'], axis = 1)

""" Se eliminan tambien las columnas de mutaciones, ya que no sirven en este caso """
data_inibica = data_inibica[data_inibica.columns.drop(list(data_inibica.filter(regex='SNV|CNV')))]

""" Se ordenan las columnas de igual manera en el que fueron colocadas durante el proceso de entrenamiento. """
cols = data_inibica.columns.tolist()
cols = cols[0:1] + cols[4:] + cols[3:4] + cols[1:2] + cols[2:3]
data_inibica = data_inibica[cols]

#data_inibica.to_excel('inference_inibica_anatomopathologic-clinical.xlsx')

""" Se carga el Excel de nuevo ya que anteriormente se ha guardado """
data_inibica_complete = pd.read_excel('/home/avalderas/img_slides/correlations/anatomopathologic-clinical/inference/test_data&models/inference_inibica_anatomopathologic-clinical.xlsx',
                                      engine='openpyxl')

""" Ahora habria que eliminar la columna de pacientes y dividir las columnas en entradas y salidas """
data_inibica_complete = data_inibica_complete.drop(['Paciente'], axis = 1)

test_tabular_data = data_inibica_complete.iloc[:, :-3]

test_labels_metastasis = data_inibica_complete.iloc[:, -1]
test_labels_survival = data_inibica_complete.iloc[:, -3]
test_labels_relapse = data_inibica_complete.iloc[:, -2]

""" Para poder realizar la inferencia hace falta transformar los dataframes en arrays de numpy. """
test_tabular_data = np.asarray(test_tabular_data).astype('float32')

test_labels_metastasis = np.asarray(test_labels_metastasis).astype('float32')
test_labels_survival = np.asarray(test_labels_survival).astype('float32')
test_labels_relapse = np.asarray(test_labels_relapse).astype('float32')

""" Una vez ya se tienen las entradas y las tres salidas correctamente en formato numpy, se carga el modelo de red para
realizar la inferencia. """
model = load_model('/home/avalderas/img_slides/correlations/anatomopathologic-clinical/inference/test_data&models/anatomopathologic-clinical.h5')

""" Se evalua los pacientes del INiBICA con los datos de test y se obtienen los resultados de las distintas métricas. """
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