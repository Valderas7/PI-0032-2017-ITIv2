import shelve # datos persistentes
import pandas as pd
import numpy as np
import imblearn
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
import itertools
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import * # Para instanciar tensores de Keras
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.model_selection import train_test_split # Se importa la librería para dividir los datos en entreno y test.
from sklearn.preprocessing import MinMaxScaler # Para escalar valores
from sklearn.metrics import confusion_matrix # Para realizar la matriz de confusión

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN DATOS TABULARES ------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# 1) Datos clínicos: Edad diagnóstico, neoadyuvante, antecedentes, metástasis a distancia, supervivencia, recaída.
# 2) Datos de anatomía patológica: Tipo histológico, STAGE, pT, pN, pM, IHQ.
list_to_read = ['CNV_oncomine', 'age', 'all_oncomine', 'mutations_oncomine', 'cancer_type', 'cancer_type_detailed',
                'dfs_months', 'dfs_status', 'dict_genes', 'dss_months', 'dss_status', 'ethnicity',
                'full_length_oncomine', 'fusions_oncomine', 'muted_genes', 'CNA_genes', 'hotspot_oncomine', 'mutations',
                'CNAs', 'neoadjuvant', 'os_months', 'os_status', 'path_m_stage', 'path_n_stage', 'path_t_stage', 'sex',
                'stage', 'subtype', 'tumor_type', 'new_tumor', 'person_neoplasm_status', 'prior_diagnosis',
                'pfs_months', 'pfs_status', 'radiation_therapy']

filename = '/home/avalderas/img_slides/data/brca_tcga_pan_can_atlas_2018.out'
#filename = 'C:\\Users\\valde\Desktop\Datos_repositorio\\tcga_data\data/brca_tcga_pan_can_atlas_2018.out'

""" Almacenamos en una variable los diccionarios: """
with shelve.open(filename) as data:
    dict_genes = data.get('dict_genes')
    age = data.get('age')
    cancer_type_detailed = data.get('cancer_type_detailed')
    neoadjuvant = data.get('neoadjuvant')  # 1 valor nulo
    os_months = data.get('os_months')
    os_status = data.get('os_status')
    path_m_stage = data.get('path_m_stage')
    path_n_stage = data.get('path_n_stage')
    path_t_stage = data.get('path_t_stage')
    stage = data.get('stage')
    subtype = data.get('subtype')
    tumor_type = data.get('tumor_type')
    new_tumor = data.get('new_tumor')  # ~200 valores nulos
    prior_diagnosis = data.get('prior_diagnosis')  # 1 valor nulo
    pfs_months = data.get('pfs_months')  # 2 valores nulos
    pfs_status = data.get('pfs_status')  # 1 valor nulo
    dfs_months = data.get('dfs_months')  # 143 valores nulos
    dfs_status = data.get('dfs_status')  # 142 valores nulos
    cnv = data.get('CNAs')
    snv = data.get('mutations')

""" Se crean dataframes individuales para cada uno de los diccionarios almacenados en cada variable de entrada y se
renombran las columnas para que todo quede más claro. Además, se crea una lista con todos los dataframes para
posteriormente unirlos todos juntos. """
df_age = pd.DataFrame.from_dict(age.items()); df_age.rename(columns = {0 : 'ID', 1 : 'Age'}, inplace = True)
df_cancer_type_detailed = pd.DataFrame.from_dict(cancer_type_detailed.items()); df_cancer_type_detailed.rename(columns = {0 : 'ID', 1 : 'cancer_type_detailed'}, inplace = True)
df_neoadjuvant = pd.DataFrame.from_dict(neoadjuvant.items()); df_neoadjuvant.rename(columns = {0 : 'ID', 1 : 'neoadjuvant'}, inplace = True)
df_dfs_status = pd.DataFrame.from_dict(dfs_status.items()); df_dfs_status.rename(columns = {0 : 'ID', 1 : 'dfs_status'}, inplace = True)
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_path_t_stage = pd.DataFrame.from_dict(path_t_stage.items()); df_path_t_stage.rename(columns = {0 : 'ID', 1 : 'path_t_stage'}, inplace = True)
df_stage = pd.DataFrame.from_dict(stage.items()); df_stage.rename(columns = {0 : 'ID', 1 : 'stage'}, inplace = True)
df_subtype = pd.DataFrame.from_dict(subtype.items()); df_subtype.rename(columns = {0 : 'ID', 1 : 'subtype'}, inplace = True)
df_tumor_type = pd.DataFrame.from_dict(tumor_type.items()); df_tumor_type.rename(columns = {0 : 'ID', 1 : 'tumor_type'}, inplace = True)
df_prior_diagnosis = pd.DataFrame.from_dict(prior_diagnosis.items()); df_prior_diagnosis.rename(columns = {0 : 'ID', 1 : 'prior_diagnosis'}, inplace = True)
df_snv = pd.DataFrame.from_dict(snv.items()); df_snv.rename(columns = {0 : 'ID', 1 : 'SNV'}, inplace = True)
df_cnv = pd.DataFrame.from_dict(cnv.items()); df_cnv.rename(columns = {0 : 'ID', 1 : 'CNV'}, inplace = True)
df_os_status = pd.DataFrame.from_dict(os_status.items()); df_os_status.rename(columns = {0 : 'ID', 1 : 'os_status'}, inplace = True)

df_path_m_stage = pd.DataFrame.from_dict(path_m_stage.items()); df_path_m_stage.rename(columns = {0 : 'ID', 1 : 'path_m_stage'}, inplace = True)

df_list = [df_age, df_neoadjuvant, df_prior_diagnosis, df_os_status, df_dfs_status, df_tumor_type, df_path_m_stage,
           df_path_n_stage, df_path_t_stage, df_stage, df_subtype]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

""" En este caso, se eliminan los pacientes con categoria 'N0 o 'NX', aquellos pacientes a los que no se les puede 
determinar si tienen o no metastasis. """
df_all_merge = df_all_merge[(df_all_merge["path_n_stage"]!='N0') & (df_all_merge["path_n_stage"]!='NX') &
                            (df_all_merge["path_n_stage"]!='N0 (I-)') & (df_all_merge["path_n_stage"]!='N0 (I+)') &
                            (df_all_merge["path_n_stage"]!='N0 (MOL+)')]

""" Se convierten las columnas de pocos valores en columnas binarias: """
df_all_merge.loc[df_all_merge.tumor_type == "Infiltrating Carcinoma (NOS)", "tumor_type"] = "Mixed Histology (NOS)"
df_all_merge.loc[df_all_merge.tumor_type == "Breast Invasive Carcinoma", "tumor_type"] = "Infiltrating Ductal Carcinoma"
df_all_merge.loc[df_all_merge.neoadjuvant == "No", "neoadjuvant"] = 0; df_all_merge.loc[df_all_merge.neoadjuvant == "Yes", "neoadjuvant"] = 1
df_all_merge.loc[df_all_merge.prior_diagnosis == "No", "prior_diagnosis"] = 0; df_all_merge.loc[df_all_merge.prior_diagnosis == "Yes", "prior_diagnosis"] = 1
df_all_merge.loc[df_all_merge.path_m_stage == "CM0 (I+)", "path_m_stage"] = "M0"
df_all_merge.loc[df_all_merge.os_status == "0:LIVING", "os_status"] = 0; df_all_merge.loc[df_all_merge.os_status == "1:DECEASED", "os_status"] = 1
df_all_merge.loc[df_all_merge.dfs_status == "0:DiseaseFree", "dfs_status"] = 0; df_all_merge.loc[df_all_merge.dfs_status == "1:Recurred/Progressed", "dfs_status"] = 1

""" Se crea una nueva columna para indicar la metastasis a distancia. En esta columna se indicaran los pacientes que 
tienen estadio M1 (metastasis inicial) + otros pacientes que desarrollan metastasis a lo largo de la enfermedad (para
ello se hace uso del excel pacientes_tcga y su columna DB) """
df_all_merge['distant_metastasis'] = 0
df_all_merge.loc[df_all_merge.path_m_stage == 'M1', 'distant_metastasis'] = 1

""" Estos pacientes desarrollan metastasis A LO LARGO de la enfermedad, tal y como se puede apreciar en el excel de los
pacientes de TCGA. Por tanto, se incluyen como clase positiva dentro de la columna 'distant_metastasis'. """
df_all_merge.loc[df_all_merge.ID == 'TCGA-A2-A3XS', 'distant_metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-AC-A2FM', 'distant_metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-AR-A2LH', 'distant_metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-BH-A0C1', 'distant_metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-BH-A18V', 'distant_metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-EW-A1P8', 'distant_metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-GM-A2DA', 'distant_metastasis'] = 1

""" Ahora, antes de transformar las variables categóricas en numéricas, se eliminan las filas donde haya datos nulos
para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Una vez la tabla tiene las columnas deseadas se procede a codificar las columnas categóricas del dataframe a valores
numéricos mediante la técnica del 'One Hot Encoding'. Más adelante se escalarán las columnas numéricas continuas, pero
ahora se realiza esta técnica antes de hacer la repartición de subconjuntos para que no haya problemas con las columnas. """
#@ get_dummies: Aplica técnica de 'One Hot Encoding', creando columnas binarias para las columnas seleccionadas
df_all_merge = pd.get_dummies(df_all_merge, columns=["tumor_type", "stage", "path_t_stage", "path_n_stage", "path_m_stage",
                                                     "subtype"])

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
# 298train, 34val, 84test
train_tabular_data, test_tabular_data = train_test_split(df_all_merge, test_size = 0.20)
train_tabular_data, valid_tabular_data = train_test_split(train_tabular_data, test_size = 0.10)

""" Ya se puede eliminar de los dos subconjuntos la columna 'ID' que no es útil para la red MLP: """
train_tabular_data = train_tabular_data.drop(['ID'], axis=1)
valid_tabular_data = valid_tabular_data.drop(['ID'], axis=1)
test_tabular_data = test_tabular_data.drop(['ID'], axis=1)

""" Se dividen los datos clínicos y los datos de anatomía patológica en datos de entrada y datos de salida."""
train_labels_tumor_type = train_tabular_data.iloc[:, 6:12]
valid_labels_tumor_type = valid_tabular_data.iloc[:, 6:12]
test_labels_tumor_type = test_tabular_data.iloc[:, 6:12]

train_labels_STAGE = train_tabular_data.iloc[:, 12:21]
valid_labels_STAGE = valid_tabular_data.iloc[:, 12:21]
test_labels_STAGE = test_tabular_data.iloc[:, 12:21]

train_labels_pT = train_tabular_data.iloc[:, 21:30]
valid_labels_pT = valid_tabular_data.iloc[:, 21:30]
test_labels_pT = test_tabular_data.iloc[:, 21:30]

train_labels_pN = train_tabular_data.iloc[:, 30:40]
valid_labels_pN = valid_tabular_data.iloc[:, 30:40]
test_labels_pN = test_tabular_data.iloc[:, 30:40]

train_labels_pM = train_tabular_data.iloc[:, 40:43]
valid_labels_pM = valid_tabular_data.iloc[:, 40:43]
test_labels_pM = test_tabular_data.iloc[:, 40:43]

train_labels_IHQ = train_tabular_data.iloc[:, 43:]
valid_labels_IHQ = valid_tabular_data.iloc[:, 43:]
test_labels_IHQ = test_tabular_data.iloc[:, 43:]

train_tabular_data = train_tabular_data.iloc[:, :6]
valid_tabular_data = valid_tabular_data.iloc[:, :6]
test_tabular_data = test_tabular_data.iloc[:, :6]

""" Se extraen los nombres de las columnas de las clases de salida para usarlos posteriormente en la sección de
evaluación: """
test_columns_tumor_type = test_labels_tumor_type.columns.values
classes_tumor_type = test_columns_tumor_type.tolist()

test_columns_STAGE = test_labels_STAGE.columns.values
classes_STAGE = test_columns_STAGE.tolist()

test_columns_pT = test_labels_pT.columns.values
classes_pT = test_columns_pT.tolist()

test_columns_pN = test_labels_pN.columns.values
classes_pN = test_columns_pN.tolist()

test_columns_pM = test_labels_pM.columns.values
classes_pM = test_columns_pM.tolist()

test_columns_IHQ = test_labels_IHQ.columns.values
classes_IHQ = test_columns_IHQ.tolist()

""" Ahora se procede a procesar las columnas continuas, que se escalarán para que estén en el rango de (0-1), es decir, 
como la salida de la red. """
scaler = MinMaxScaler()

""" Hay 'warning' si se hace directamente, así que se hace de esta manera. Se transforman los datos guardándolos en una
variable. Posteriormente se modifica la columna de las tablas con esa variable. """
train_continuous = scaler.fit_transform(train_tabular_data[['Age']])
valid_continuous = scaler.transform(valid_tabular_data[['Age']])
test_continuous = scaler.transform(test_tabular_data[['Age']])

train_tabular_data.loc[:,'Age'] = train_continuous[:,0]
valid_tabular_data.loc[:,'Age'] = valid_continuous[:,0]
test_tabular_data.loc[:,'Age'] = test_continuous[:,0]

""" Para poder entrenar la red hace falta transformar los dataframes de entrenamiento y test en arrays de numpy. """
train_tabular_data = np.asarray(train_tabular_data).astype('float32')
train_labels_tumor_type = np.asarray(train_labels_tumor_type).astype('float32')
train_labels_STAGE = np.asarray(train_labels_STAGE).astype('float32')
train_labels_pT = np.asarray(train_labels_pT).astype('float32')
train_labels_pN = np.asarray(train_labels_pN).astype('float32')
train_labels_pM = np.asarray(train_labels_pM).astype('float32')
train_labels_IHQ = np.asarray(train_labels_IHQ).astype('float32')

valid_tabular_data = np.asarray(valid_tabular_data).astype('float32')
valid_labels_tumor_type = np.asarray(valid_labels_tumor_type).astype('float32')
valid_labels_STAGE = np.asarray(valid_labels_STAGE).astype('float32')
valid_labels_pT = np.asarray(valid_labels_pT).astype('float32')
valid_labels_pN = np.asarray(valid_labels_pN).astype('float32')
valid_labels_pM = np.asarray(valid_labels_pM).astype('float32')
valid_labels_IHQ = np.asarray(valid_labels_IHQ).astype('float32')

test_tabular_data = np.asarray(test_tabular_data).astype('float32')
test_labels_tumor_type = np.asarray(test_labels_tumor_type).astype('float32')
test_labels_STAGE = np.asarray(test_labels_STAGE).astype('float32')
test_labels_pT = np.asarray(test_labels_pT).astype('float32')
test_labels_pN = np.asarray(test_labels_pN).astype('float32')
test_labels_pM = np.asarray(test_labels_pM).astype('float32')
test_labels_IHQ = np.asarray(test_labels_IHQ).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (MLP) -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
Input_ = Input(shape=train_tabular_data.shape[1], )
model = layers.Dense(128, activation= 'relu')(Input_)
model = layers.Dropout(0.5)(model)
model = layers.Dense(64, activation= 'relu')(model)
model = layers.Dropout(0.5)(model)
output1 = layers.Dense(train_labels_tumor_type.shape[1], activation = "softmax", name= 'tumor_type')(model)
output2 = layers.Dense(train_labels_STAGE.shape[1], activation = "softmax", name = 'STAGE')(model)
output3 = layers.Dense(train_labels_pT.shape[1], activation = "softmax", name= 'pT')(model)
output4 = layers.Dense(train_labels_pN.shape[1], activation = "softmax", name= 'pN')(model)
output5 = layers.Dense(train_labels_pM.shape[1], activation = "softmax", name= 'pM')(model)
output6 = layers.Dense(train_labels_IHQ.shape[1], activation = "softmax", name= 'IHQ')(model)

model = Model(inputs = Input_, outputs = [output1, output2, output3, output4, output5, output6])

""" Hay que definir las métricas de la red y configurar los distintos hiperparámetros para entrenar la red. El modelo ya
ha sido definido anteriormente, así que ahora hay que compilarlo. Para ello se define una función de loss y un 
optimizador. Con la función de loss se estimará la 'loss' del modelo. Por su parte, el optimizador actualizará los
parámetros de la red neuronal con el objetivo de minimizar la función de 'loss'. """
# @lr: tamaño de pasos para alcanzar el mínimo global de la función de loss.
metrics = [keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'),
           keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn'),
           keras.metrics.Recall(name='recall'), # TP / (TP + FN)
           keras.metrics.Precision(name='precision'), # TP / (TP + FP)
           keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC')]

model.compile(loss = {'tumor_type': 'categorical_crossentropy', 'STAGE': 'categorical_crossentropy',
                      'pT': 'categorical_crossentropy', 'pN': 'categorical_crossentropy',
                      'pM': 'categorical_crossentropy', 'IHQ': 'categorical_crossentropy'},
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)

model.summary()

""" Se implementa un callback: para guardar el mejor modelo que tenga la menor 'loss' en la validación. """
checkpoint_path = '/home/avalderas/img_slides/correlations/clinical-anatomopathologic/inference/test_data&models/clinical-anatomopathologic.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = True, monitor= 'loss', mode= 'min')

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
neural_network = model.fit(x = train_tabular_data,
                           y = {'tumor_type': train_labels_tumor_type, 'STAGE': train_labels_STAGE,
                                'pT': train_labels_pT, 'pN': train_labels_pN, 'pM': train_labels_pM,
                                'IHQ': train_labels_IHQ},
                           epochs = 1000,
                           verbose = 1,
                           batch_size = 32,
                           #callbacks= mcp_save,
                           validation_data = (valid_tabular_data, {'tumor_type': valid_labels_tumor_type,
                                                                   'STAGE': valid_labels_STAGE, 'pT': valid_labels_pT,
                                                                   'pN': valid_labels_pN, 'pM': valid_labels_pM,
                                                                   'IHQ': valid_labels_IHQ}))

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

"""Las métricas del entreno se guardan dentro del método 'history'. Primero, se definen las variables para usarlas 
posteriormentes para dibujar las gráficas de la 'loss', la sensibilidad y la precisión del entrenamiento y  validación 
de cada iteración."""
loss = neural_network.history['loss']
val_loss = neural_network.history['val_loss']

epochs = neural_network.epoch

""" Una vez definidas las variables se dibujan las distintas gráficas. """
""" Gráfica de la 'loss' del entreno y la validación: """
plt.plot(epochs, loss, 'r', label='Loss del entreno')
plt.plot(epochs, val_loss, 'b--', label='Loss de la validación')
plt.title('Loss del entreno y de la validación')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.figure() # Crea o activa una figura
plt.show() # Se muestran todas las gráficas

""" -------------------------------------------------------------------------------------------------------------------
------------------------------------------- SECCIÓN DE EVALUACIÓN  ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos. """
# @suppress = True: Muestra los números con representación de coma fija
# @predict: Genera predicciones para nuevas entradas
#print("\nGenera predicciones para 10 muestras")
#print("Clase del primer paciente: \n", test_labels_tumor_type[:1], test_labels_STAGE[:1], test_labels_pT[:1],
      #test_labels_pN[:1], test_labels_pM[:1], test_labels_IHQ[:1])
np.set_printoptions(precision=3, suppress=True)
#print("\nPredicciones:\n", np.round(model.predict(test_tabular_data[:1])[0])) # El índice da una salida u otra

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
# Tipo histológico
y_true_tumor_type = []
for label_test_tumor_type in test_labels_tumor_type:
    y_true_tumor_type.append(np.argmax(label_test_tumor_type))

y_true_tumor_type = np.array(y_true_tumor_type)
y_pred_tumor_type = np.argmax(model.predict(test_tabular_data)[0], axis = 1)

matrix_tumor_type = confusion_matrix(y_true_tumor_type, y_pred_tumor_type) # Calcula (pero no dibuja) la matriz de confusión
matrix_tumor_type_classes = ['IDC', 'ILC', 'Metaplastic', 'Mixed (NOS)', 'Mucinous', 'Other']

# Estadio anatomopatológico
y_true_STAGE = []
for label_test_STAGE in test_labels_STAGE:
    y_true_STAGE.append(np.argmax(label_test_STAGE))

y_true_STAGE = np.array(y_true_STAGE)
y_pred_STAGE = np.argmax(model.predict(test_tabular_data)[1], axis = 1)

matrix_STAGE = confusion_matrix(y_true_STAGE, y_pred_STAGE) # Calcula (pero no dibuja) la matriz de confusión
matrix_STAGE_classes = ['Stage IB', 'Stage II', 'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IIIA', 'Stage IIIB',
                        'Stage IIIC', 'STAGE X']

# pT
y_true_pT = []
for label_test_pT in test_labels_pT:
    y_true_pT.append(np.argmax(label_test_pT))

y_true_pT = np.array(y_true_pT)
y_pred_pT = np.argmax(model.predict(test_tabular_data)[2], axis = 1)

matrix_pT = confusion_matrix(y_true_pT, y_pred_pT) # Calcula (pero no dibuja) la matriz de confusión
matrix_pT_classes = ['T1', 'T1B', 'T1C', 'T2', 'T2B', 'T3', 'T4', 'T4B', 'T4D']

# pN
y_true_pN = []
for label_test_pN in test_labels_pN:
    y_true_pN.append(np.argmax(label_test_pN))

y_true_pN = np.array(y_true_pN)
y_pred_pN = np.argmax(model.predict(test_tabular_data)[3], axis = 1)

matrix_pN = confusion_matrix(y_true_pN, y_pred_pN) # Calcula (pero no dibuja) la matriz de confusión
matrix_pN_classes = ['N1', 'N1A', 'N1B', 'N1C', 'N1MI', 'N2', 'N2A', 'N3', 'N3A', 'N3B']

# pM
y_true_pM = []
for label_test_pM in test_labels_pM:
    y_true_pM.append(np.argmax(label_test_pM))

y_true_pM = np.array(y_true_pM)
y_pred_pM = np.argmax(model.predict(test_tabular_data)[4], axis = 1)

matrix_pM = confusion_matrix(y_true_pM, y_pred_pM) # Calcula (pero no dibuja) la matriz de confusión
matrix_pM_classes = ['M0', 'M1', 'MX']

# IHQ
y_true_IHQ = []
for label_test_IHQ in test_labels_IHQ:
    y_true_IHQ.append(np.argmax(label_test_IHQ))

y_true_IHQ = np.array(y_true_IHQ)
y_pred_IHQ = np.argmax(model.predict(test_tabular_data)[5], axis = 1)

matrix_IHQ = confusion_matrix(y_true_IHQ, y_pred_IHQ) # Calcula (pero no dibuja) la matriz de confusión
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

plot_confusion_matrix(matrix_tumor_type, classes = matrix_tumor_type_classes, title = 'Matriz de confusión tipo '
                                                                                      'histológico')
plt.show()

plot_confusion_matrix(matrix_STAGE, classes = matrix_STAGE_classes, title = 'Matriz del estadio anatomopatológico')
plt.show()

plot_confusion_matrix(matrix_pT, classes = matrix_pT_classes, title = 'Matriz del parámetro "T"')
plt.show()

plot_confusion_matrix(matrix_pN, classes = matrix_pN_classes, title = 'Matriz del parámetro "N"')
plt.show()

plot_confusion_matrix(matrix_pM, classes = matrix_pM_classes, title = 'Matriz del parámetro "M"')
plt.show()

plot_confusion_matrix(matrix_IHQ, classes = matrix_IHQ_classes, title = 'Matriz del subtipo molecular')
plt.show()

#np.save('test_data', test_tabular_data)
#np.save('test_labels_tumor_type', test_labels_tumor_type)
#np.save('test_labels_STAGE', test_labels_STAGE)
#np.save('test_labels_pT', test_labels_pT)
#np.save('test_labels_pN', test_labels_pN)
#np.save('test_labels_pM', test_labels_pM)
#np.save('test_labels_IHQ', test_labels_IHQ)