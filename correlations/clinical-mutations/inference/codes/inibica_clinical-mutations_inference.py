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
from sklearn.metrics import multilabel_confusion_matrix
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

""" Se eliminan las columnas de anatomía patológica, ya que no sirven en este caso, además de las mutaciones CNV de tipo 
'NORMAL' """
data_inibica = data_inibica.drop(['ER', 'PR', 'Ki-67', 'Her-2', 'Tipo_tumor', 'STAGE', 'pT', 'pN', 'pM'], axis = 1)
data_inibica = data_inibica[data_inibica.columns.drop(list(data_inibica.filter(regex='NORMAL')))]

""" Se ordenan las columnas de igual manera en el que fueron colocadas durante el proceso de entrenamiento. """
cols = data_inibica.columns.tolist()
cols = cols[:2] + cols[6:7] + cols[2:3] + cols[5:6] + cols[3:5] + cols[7:]
data_inibica = data_inibica[cols]

#data_inibica.to_excel('inference_inibica_clinical-mutations.xlsx')

""" Se carga el Excel de nuevo ya que anteriormente se ha guardado """
data_inibica_complete = pd.read_excel('/home/avalderas/img_slides/correlations/clinical-mutations/inference/test_data&models/inference_inibica_clinical-mutations.xlsx',
                                      engine='openpyxl')

""" Ahora habría que eliminar la columna de pacientes y dividir las columnas en entradas (datos clínicos) y salidas 
(SNV, CNV-A, CNV-D). """
data_inibica_complete = data_inibica_complete.drop(['Paciente'], axis = 1)

test_tabular_data = data_inibica_complete.iloc[:, :6]

test_labels_snv = data_inibica_complete.iloc[:, 6:157]
test_labels_cnv_a = data_inibica_complete.iloc[:, 157::2]
test_labels_cnv_d = data_inibica_complete.iloc[:, 158::2]

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
model = load_model('/correlations/clinical-mutations/inference/test_data/clinical-mutations.h5')

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
    #plt.savefig('/home/avalderas/img_slides/screenshots/correlations/clinical - mutations/inibica/SNV/{}'.format(classes_snv[index_snv].split('_')[1]), bbox_inches='tight')
    plt.show()
    plt.pause(0.2)
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
    #plt.savefig('/home/avalderas/img_slides/screenshots/correlations/clinical - mutations/inibica/CNV-A/{}'.format(classes_cnv_a[index_cnv_a].split('_')[1]), bbox_inches='tight')
    plt.show()
    plt.pause(0.2)
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
    #plt.savefig('/home/avalderas/img_slides/screenshots/correlations/clinical - mutations/inibica/CNV-D/{}'.format(classes_cnv_d[index_cnv_d].split('_')[1]), bbox_inches='tight')
    plt.show()
    plt.pause(0.2)
    index_cnv_d = index_cnv_d + 1