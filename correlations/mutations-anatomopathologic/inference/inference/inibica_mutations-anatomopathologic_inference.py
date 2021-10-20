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
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.metrics import confusion_matrix

""" Se carga el Excel de los pacientes de INiBICA """
data_inibica = pd.read_excel('/home/avalderas/img_slides/excel_genesOCA&inibica_patients/inference_inibica.xlsx',
                             engine='openpyxl')

""" Se crea la misma cantidad de columnas para los tipos de tumor que se creo en el conjunto de entrenamiento para 
rellenarlas posteriormente con un '1' en las filas que corresponda, dependiendo de los tipos de tumor de los pacientes 
de INiBICA. """
data_inibica['Tumor_IDC'] = 0
data_inibica['Tumor_ILC'] = 0
data_inibica['Tumor_Medullary'] = 0
data_inibica['Tumor_Metaplastic'] = 0
data_inibica['Tumor_Mixed'] = 0
data_inibica['Tumor_Mucinous'] = 0
data_inibica['Tumor_Other'] = 0

data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Medullary'), 'Tumor_Medullary'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Mucinous'), 'Tumor_Mucinous'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'].str.contains("Lobular")) |
                 (data_inibica['Tipo_tumor'] == 'Signet-ring cells lobular'), 'Tumor_ILC'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Invasive carcinoma (NST)') |
                 (data_inibica['Tipo_tumor'] == 'Microinvasive carcinoma') |
                 (data_inibica['Tipo_tumor'] == 'Signet-ring cells ductal'), 'Tumor_IDC'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Mixed ductal and lobular'), 'Tumor_Mixed'] = 1
data_inibica.loc[(data_inibica['Tipo_tumor'] == 'Apocrine') | (data_inibica['Tipo_tumor'] == 'Papillary') |
                 (data_inibica['Tipo_tumor'] == 'Tubular'), 'Tumor_Other'] = 1

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
data_inibica['STAGE_IV'] = 0
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
data_inibica['pT_T1A'] = 0
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
data_inibica['pN_N3C'] = 0

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

""" Se eliminan las columnas de variables clínicas, ya que no sirven en este caso, y las columnas 'CNV' de tipo 'NORMAL' """
data_inibica = data_inibica.drop(['Edad', 'Diagnóstico_previo', 'Recidivas', 'Metástasis_distancia',
                                  'Estado_supervivencia', 'Tratamiento_neoadyuvante'], axis = 1)
data_inibica = data_inibica[data_inibica.columns.drop(list(data_inibica.filter(regex='NORMAL')))]

""" Se ordenan las columnas de igual manera en el que fueron colocadas durante el proceso de entrenamiento. """
cols = data_inibica.columns.tolist()
cols = cols[:1] + cols[-46:] + cols[1:-46]
data_inibica = data_inibica[cols]

#data_inibica.to_excel('inference_inibica_anatomo-mutations.xlsx')

""" Se carga el Excel de nuevo ya que anteriormente se ha guardado """
data_inibica_complete = pd.read_excel('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/inference_inibica_mutations-anatomopathologic.xlsx',
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
model = load_model('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/mutations-anatomopathologic.h5')