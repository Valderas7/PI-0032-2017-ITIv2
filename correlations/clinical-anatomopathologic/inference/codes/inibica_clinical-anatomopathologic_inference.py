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

""" Se aprecian valores nulos en la columna de 'Diagnostico Previo'. Para nos desechar estos pacientes, se ha optado por 
considerarlos como pacientes sin diagnostico previo: """
data_inibica['Diagnóstico_previo'].fillna(value = 0, inplace= True)

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

""" Se eliminan las columnas de IHQ, ya que no nos sirven """
data_inibica = data_inibica.drop(['ER', 'PR', 'Ki-67', 'Her-2'], axis = 1)

""" Se eliminan las columnas de mutaciones, ya que no sirven en este caso """
data_inibica = data_inibica[data_inibica.columns.drop(list(data_inibica.filter(regex='SNV|CNV')))]

""" Se ordenan las columnas de igual manera en el que fueron colocadas durante el proceso de entrenamiento. """
cols = data_inibica.columns.tolist()
cols = cols[:2] + cols[6:7] + cols[2:3] + cols[5:6] + cols[3:5] + cols[7:]
data_inibica = data_inibica[cols]

#data_inibica.to_excel('inference_inibica_clinical-anatomopathologic.xlsx')

""" Se carga el Excel de nuevo ya que anteriormente se ha guardado """
data_inibica_complete = pd.read_excel('/home/avalderas/img_slides/correlations/clinical-anatomopathologic/inference/test_data&models/inference_inibica_clinical-anatomopathologic.xlsx',
                                      engine='openpyxl')

""" Ahora habria que eliminar la columna de pacientes y dividir las columnas en entradas y salidas de tipo histológico,
estadio anatomopatológico, pT, pN, pM e IHQ. """
data_inibica_complete = data_inibica_complete.drop(['Paciente'], axis = 1)

test_tabular_data = data_inibica_complete.iloc[:, :6]

test_labels_tumor_type = data_inibica_complete.iloc[:, 6:12]
test_labels_STAGE = data_inibica_complete.iloc[:, 12:21]
test_labels_pT = data_inibica_complete.iloc[:, 21:30]
test_labels_pN = data_inibica_complete.iloc[:, 30:40]
test_labels_pM = data_inibica_complete.iloc[:, 40:43]
test_labels_IHQ = data_inibica_complete.iloc[:, 43:]

""" Para poder realizar la inferencia hace falta transformar los dataframes en arrays de numpy. """
test_tabular_data = np.asarray(test_tabular_data).astype('float32')

test_labels_tumor_type = np.asarray(test_labels_tumor_type).astype('float32')
test_labels_STAGE = np.asarray(test_labels_STAGE).astype('float32')
test_labels_pT = np.asarray(test_labels_pT).astype('float32')
test_labels_pN = np.asarray(test_labels_pN).astype('float32')
test_labels_pM = np.asarray(test_labels_pM).astype('float32')
test_labels_IHQ = np.asarray(test_labels_IHQ).astype('float32')

""" Una vez ya se tienen las entradas y las tres salidas correctamente en formato numpy, se carga el modelo de red para
realizar la inferencia. """
model = load_model('/correlations/clinical-anatomopathologic/inference/models/clinical-anatomopathologic.h5')

""" Se evalua los pacientes del INiBICA con los datos de test y se obtienen los resultados de las distintas métricas. """
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

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
# Tipo histológico
y_true_tumor_type = []
for label_test_tumor_type in test_labels_tumor_type:
    y_true_tumor_type.append(np.argmax(label_test_tumor_type))

y_true_tumor_type = np.array(y_true_tumor_type)
y_pred_tumor_type = np.argmax(model.predict(test_tabular_data)[0], axis = 1)

matrix_tumor_type = confusion_matrix(y_true_tumor_type, y_pred_tumor_type, labels = [0, 1, 2, 3, 4, 5]) # Calcula (pero no dibuja) la matriz de confusión
matrix_tumor_type_classes = ['IDC', 'ILC', 'Metaplastic', 'Mixed (NOS)', 'Mucinous', 'Other']

# Estadio anatomopatológico
y_true_STAGE = []
for label_test_STAGE in test_labels_STAGE:
    y_true_STAGE.append(np.argmax(label_test_STAGE))

y_true_STAGE = np.array(y_true_STAGE)
y_pred_STAGE = np.argmax(model.predict(test_tabular_data)[1], axis = 1)

matrix_STAGE = confusion_matrix(y_true_STAGE, y_pred_STAGE, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]) # Calcula (pero no dibuja) la matriz de confusión
matrix_STAGE_classes = ['Stage IB', 'Stage II', 'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IIIA', 'Stage IIIB',
                        'Stage IIIC', 'STAGE X']

# pT
y_true_pT = []
for label_test_pT in test_labels_pT:
    y_true_pT.append(np.argmax(label_test_pT))

y_true_pT = np.array(y_true_pT)
y_pred_pT = np.argmax(model.predict(test_tabular_data)[2], axis = 1)

matrix_pT = confusion_matrix(y_true_pT, y_pred_pT, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pT_classes = ['T1', 'T1B', 'T1C', 'T2', 'T2B', 'T3', 'T4', 'T4B', 'T4D']

# pN
y_true_pN = []
for label_test_pN in test_labels_pN:
    y_true_pN.append(np.argmax(label_test_pN))

y_true_pN = np.array(y_true_pN)
y_pred_pN = np.argmax(model.predict(test_tabular_data)[3], axis = 1)

matrix_pN = confusion_matrix(y_true_pN, y_pred_pN, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pN_classes = ['N1', 'N1A', 'N1B', 'N1C', 'N1MI', 'N2', 'N2A', 'N3', 'N3A', 'N3B']

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
fig1 = plt.figure()

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