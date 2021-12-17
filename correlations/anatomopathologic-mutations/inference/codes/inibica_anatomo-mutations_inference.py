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
from sklearn.metrics import multilabel_confusion_matrix

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
data_inibica_complete = pd.read_excel('/home/avalderas/img_slides/correlations/anatomopathologic-mutations/inference/test_data&models/inference_inibica_anatomo-mutations.xlsx',
                                      engine='openpyxl')

""" Ahora habria que eliminar la columna de pacientes y dividir las columnas en entradas y salidas de snv, cn-a y cnv-d. """
data_inibica_complete = data_inibica_complete.drop(['Paciente'], axis = 1)

test_tabular_data = data_inibica_complete.iloc[:, :46]

test_labels_snv = data_inibica_complete.iloc[:, 46:197]
test_labels_cnv_a = data_inibica_complete.iloc[:, 197::2]
test_labels_cnv_d = data_inibica_complete.iloc[:, 198::2]

""" Se extraen los nombres de las columnas de las clases de salida para usarlos posteriormente en la sección de
evaluación: """
test_columns_snv = test_labels_snv.columns.values
classes_snv = test_columns_snv.tolist()

test_columns_cnv_a = test_labels_cnv_a.columns.values
classes_cnv_a = test_columns_cnv_a.tolist()

test_columns_cnv_d = test_labels_cnv_d.columns.values
classes_cnv_d = test_columns_cnv_d.tolist()

""" Para poder realizar la inferencia hace falta transformar los dataframes en arrays de numpy. """
test_tabular_data = np.asarray(test_tabular_data).astype('float32')

test_labels_snv = np.asarray(test_labels_snv).astype('float32')
test_labels_cnv_a = np.asarray(test_labels_cnv_a).astype('float32')
test_labels_cnv_d = np.asarray(test_labels_cnv_d).astype('float32')

""" Una vez ya se tienen las entradas y las tres salidas correctamente en formato numpy, se carga el modelo de red para
realizar la inferencia. """
model = load_model('/correlations/anatomopathologic-mutations/inference/test_data/anatomopathologic-mutations.h5')

""" Se evalua los pacientes del INiBICA con los datos de test y se obtienen los resultados de las distintas métricas. """
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_tabular_data, [test_labels_snv, test_labels_cnv_a, test_labels_cnv_d], verbose = 0)

print("\n'Loss' de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "SNV del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones SNV del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f} \n"
      "Exactitud de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones SNV"
      " del panel OCA en el conjunto de prueba: {:.2f}".format(results[1], results[8], results[9],
                                                               results[6]/(results[6]+results[5]), results[10] * 100,
                                                               results[11]))
if results[8] > 0 or results[9] > 0:
    print("Valor-F de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[8] * results[9]) /
                                                                                                (results[8] + results[9])))

print("\n'Loss' de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones CNV-A del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n"
      "Exactitud de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones "
      "CNV-A del panel OCA en el conjunto de prueba: {:.2f}".format(results[2], results[16], results[17],
                                                                    results[14]/(results[14]+results[13]),
                                                                    results[18] * 100, results[19]))
if results[16] > 0 or results[17] > 0:
    print("Valor-F de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[16] * results[17]) /
                                                                                                  (results[16] + results[17])))

print("\n'Loss' de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones CNV-D del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n"
      "Exactitud de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones "
      "CNV-D del panel OCA en el conjunto de prueba: {:.2f}".format(results[3], results[24], results[25],
                                                                    results[22]/(results[22]+results[21]),
                                                                    results[26] * 100, results[27]))
if results[24] > 0 or results[25] > 0:
    print("Valor-F de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[24] * results[25]) /
                                                                                                  (results[24] + results[25])))

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
# SNV
y_true_snv = test_labels_snv
y_pred_snv = np.round(model.predict(test_tabular_data)[0])

matrix_snv = multilabel_confusion_matrix(y_true_snv, y_pred_snv)

group_names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
index_snv = 0

for matrix_gen_snv in matrix_snv:
    group_counts = ['{0:0.0f}'.format(value) for value in matrix_gen_snv.flatten()]  # Cantidad de casos por grupo
    true_neg_pos_neg = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
    true_neg_pos_neg = np.asarray(true_neg_pos_neg).reshape(2, 2)
    sns.heatmap(matrix_gen_snv, annot=true_neg_pos_neg, fmt='', cmap='Blues')
    plt.title('Mutación SNV del gen {}'.format(classes_snv[index_snv].split('_')[1]))
    #plt.savefig('/home/avalderas/img_slides/screenshots/correlations/anatomopathologic - mutations/inibica/SNV/{}'.format(classes_snv[index_snv].split('_')[1]), bbox_inches='tight')
    plt.show()
    plt.pause(0.1)
    index_snv = index_snv + 1

# CNV-A
y_true_cnv_a = test_labels_cnv_a
y_pred_cnv_a = np.round(model.predict(test_tabular_data)[1])

matrix_cnv_a = multilabel_confusion_matrix(y_true_cnv_a, y_pred_cnv_a) # Calcula (pero no dibuja) la matriz de confusión
index_cnv_a = 0

for matrix_gen_cnv_a in matrix_cnv_a:
    group_counts = ['{0:0.0f}'.format(value) for value in matrix_gen_cnv_a.flatten()]  # Cantidad de casos por grupo
    true_neg_pos_neg = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
    true_neg_pos_neg = np.asarray(true_neg_pos_neg).reshape(2, 2)
    sns.heatmap(matrix_gen_cnv_a, annot=true_neg_pos_neg, fmt='', cmap='Blues')
    plt.title('Mutación CNV-A del gen {}'.format(classes_cnv_a[index_cnv_a].split('_')[1]))
    #plt.savefig('/home/avalderas/img_slides/screenshots/correlations/anatomopathologic - mutations/inibica/CNV-A/{}'.format(classes_cnv_a[index_cnv_a].split('_')[1]), bbox_inches='tight')
    plt.show()
    plt.pause(0.1)
    index_cnv_a = index_cnv_a + 1

# CNV-D
y_true_cnv_d = test_labels_cnv_d
y_pred_cnv_d = np.round(model.predict(test_tabular_data)[2])

matrix_cnv_d = multilabel_confusion_matrix(y_true_cnv_d, y_pred_cnv_d) # Calcula (pero no dibuja) la matriz de confusión
index_cnv_d = 0

for matrix_gen_cnv_d in matrix_cnv_d:
    group_counts = ['{0:0.0f}'.format(value) for value in matrix_gen_cnv_d.flatten()]  # Cantidad de casos por grupo
    true_neg_pos_neg = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
    true_neg_pos_neg = np.asarray(true_neg_pos_neg).reshape(2, 2)
    sns.heatmap(matrix_gen_cnv_d, annot=true_neg_pos_neg, fmt='', cmap='Blues')
    plt.title('Mutación CNV-D del gen {}'.format(classes_cnv_d[index_cnv_d].split('_')[1]))
    #plt.savefig('/home/avalderas/img_slides/screenshots/correlations/anatomopathologic - mutations/inibica/CNV-D/{}'.format(classes_cnv_d[index_cnv_d].split('_')[1]), bbox_inches='tight')
    plt.show()
    plt.pause(0.1)
    index_cnv_d = index_cnv_d + 1

""" En caso de querer curvas ROC individuales para un gen determinado se activa esta parte del codigo:
#Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
#documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
#tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases.
#Para implementarlas, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva:
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve

y_true = test_labels_cnv_a[:, classes_cnv_a.index('CNV_ERBB2_AMP')]
y_pred_prob = model.predict(test_tabular_data)[1]
y_pred_prob = y_pred_prob[:, classes_cnv_a.index('CNV_ERBB2_AMP')].ravel()
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
auc_roc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for CNV-A of ERBB2 gene')
plt.legend(loc = 'best')
plt.show()

#Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
#del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados.
precision, recall, threshold = precision_recall_curve(y_true, y_pred_prob)
auc_pr = auc(recall, precision)

plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for CNV-A of ERBB2 gene')
plt.legend(loc = 'best')
plt.show()
"""