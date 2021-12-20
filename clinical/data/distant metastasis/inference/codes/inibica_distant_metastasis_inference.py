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

""" En el caso de predecir metastasis a distancia, como no se ha entrenado con valores como MX, se sustituyen por cero """
data_inibica.loc[data_inibica.pM == "X", "pM"] = 0

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
data_inibica = data_inibica.drop(['Tipo_tumor', 'STAGE', 'Recidivas', 'Estado_supervivencia'], axis = 1)

""" Se ordenan las columnas de igual manera en el que fueron colocadas durante el proceso de entrenamiento. """
data_inibica = data_inibica[['Paciente', 'Edad', 'Tratamiento_neoadyuvante', 'pM', 'Diagnóstico_previo', 'SNV_AKT1',
                             'SNV_AKT2', 'SNV_AKT3', 'SNV_ALK', 'SNV_AR', 'SNV_ARAF', 'SNV_AXL', 'SNV_BRAF', 'SNV_BTK',
                             'SNV_CBL', 'SNV_CCND1', 'SNV_CDK4', 'SNV_CDK6', 'SNV_CHEK2', 'SNV_CSF1R', 'SNV_CTNNB1',
                             'SNV_DDR2', 'SNV_EGFR', 'SNV_ERBB2', 'SNV_ERBB3', 'SNV_ERBB4', 'SNV_ERCC2', 'SNV_ESR1',
                             'SNV_EZH2', 'SNV_FGFR1', 'SNV_FGFR2', 'SNV_FGFR3', 'SNV_FGFR4', 'SNV_FLT3', 'SNV_FOXL2',
                             'SNV_GATA2', 'SNV_GNA11', 'SNV_GNAQ', 'SNV_GNAS', 'SNV_H3F3A', 'SNV_HIST1H3B', 'SNV_HNF1A',
                             'SNV_HRAS', 'SNV_IDH1', 'SNV_IDH2', 'SNV_JAK1', 'SNV_JAK2', 'SNV_JAK3', 'SNV_KDR',
                             'SNV_KIT', 'SNV_KNSTRN', 'SNV_KRAS', 'SNV_MAGOH', 'SNV_MAP2K1', 'SNV_MAP2K2', 'SNV_MAP2K4',
                             'SNV_MAPK1', 'SNV_MAX', 'SNV_MDM4', 'SNV_MED12', 'SNV_MET', 'SNV_MTOR', 'SNV_MYC',
                             'SNV_MYCN', 'SNV_MYD88', 'SNV_NFE2L2', 'SNV_NRAS', 'SNV_NTRK1', 'SNV_NTRK2', 'SNV_NTRK3',
                             'SNV_PDGFRA', 'SNV_PDGFRB', 'SNV_PIK3CA', 'SNV_PIK3CB', 'SNV_PPP2R1A', 'SNV_PTPN11',
                             'SNV_RAC1', 'SNV_RAF1', 'SNV_RET', 'SNV_RHEB', 'SNV_RHOA', 'SNV_ROS1', 'SNV_SF3B1',
                             'SNV_SMAD4', 'SNV_SMO', 'SNV_SPOP', 'SNV_SRC', 'SNV_STAT3', 'SNV_TERT', 'SNV_TOP1',
                             'SNV_U2AF1', 'SNV_XPO1', 'SNV_BRCA1', 'SNV_BRCA2', 'SNV_CDKN2A', 'SNV_ERG', 'SNV_ETV1',
                             'SNV_ETV4', 'SNV_ETV5', 'SNV_FGR', 'SNV_MYB', 'SNV_MYBL1', 'SNV_NF1', 'SNV_NOTCH1',
                             'SNV_NOTCH4', 'SNV_NRG1', 'SNV_NUTM1', 'SNV_PPARG', 'SNV_PRKACA', 'SNV_PRKACB', 'SNV_PTEN',
                             'SNV_RAD51B', 'SNV_RB1', 'SNV_RELA', 'SNV_RSPO2', 'SNV_RSPO3', 'SNV_ARID1A', 'SNV_ATM',
                             'SNV_ATR', 'SNV_ATRX', 'SNV_BAP1', 'SNV_CDK12', 'SNV_CDKN1B', 'SNV_CDKN2B', 'SNV_CHEK1',
                             'SNV_CREBBP', 'SNV_FANCA', 'SNV_FANCD2', 'SNV_FANCI', 'SNV_FBXW7', 'SNV_MLH1', 'SNV_MRE11',
                             'SNV_MSH2', 'SNV_MSH6', 'SNV_NBN', 'SNV_NF2', 'SNV_NOTCH2', 'SNV_NOTCH3', 'SNV_PALB2',
                             'SNV_PIK3R1', 'SNV_PMS2', 'SNV_POLE', 'SNV_PTCH1', 'SNV_RAD50', 'SNV_RAD51', 'SNV_RAD51C',
                             'SNV_RAD51D', 'SNV_RNF43', 'SNV_SETD2', 'SNV_SLX4', 'SNV_SMARCA4', 'SNV_SMARCB1',
                             'SNV_STK11', 'SNV_TP53', 'SNV_TSC1', 'SNV_TSC2', 'CNV_AKT1_AMP', 'CNV_AKT1_NORMAL',
                             'CNV_AKT1_DEL', 'CNV_AKT2_AMP', 'CNV_AKT2_NORMAL', 'CNV_AKT2_DEL', 'CNV_AKT3_AMP',
                             'CNV_AKT3_NORMAL', 'CNV_AKT3_DEL', 'CNV_ALK_AMP', 'CNV_ALK_NORMAL', 'CNV_ALK_DEL',
                             'CNV_AR_AMP', 'CNV_AR_NORMAL', 'CNV_AR_DEL', 'CNV_AXL_AMP', 'CNV_AXL_NORMAL',
                             'CNV_AXL_DEL', 'CNV_BRAF_AMP', 'CNV_BRAF_NORMAL', 'CNV_BRAF_DEL', 'CNV_CCND1_AMP',
                             'CNV_CCND1_NORMAL', 'CNV_CCND1_DEL', 'CNV_CCND2_AMP', 'CNV_CCND2_NORMAL', 'CNV_CCND2_DEL',
                             'CNV_CCND3_AMP', 'CNV_CCND3_NORMAL', 'CNV_CCND3_DEL', 'CNV_CCNE1_AMP', 'CNV_CCNE1_NORMAL',
                             'CNV_CCNE1_DEL', 'CNV_CDK2_AMP', 'CNV_CDK2_NORMAL', 'CNV_CDK2_DEL', 'CNV_CDK4_AMP',
                             'CNV_CDK4_NORMAL', 'CNV_CDK4_DEL', 'CNV_CDK6_AMP', 'CNV_CDK6_NORMAL', 'CNV_CDK6_DEL',
                             'CNV_EGFR_AMP', 'CNV_EGFR_NORMAL', 'CNV_EGFR_DEL', 'CNV_ERBB2_AMP', 'CNV_ERBB2_NORMAL',
                             'CNV_ERBB2_DEL', 'CNV_ESR1_AMP', 'CNV_ESR1_NORMAL', 'CNV_ESR1_DEL', 'CNV_FGF19_AMP',
                             'CNV_FGF19_NORMAL', 'CNV_FGF19_DEL', 'CNV_FGF3_AMP', 'CNV_FGF3_NORMAL', 'CNV_FGF3_DEL',
                             'CNV_FGFR1_AMP', 'CNV_FGFR1_NORMAL', 'CNV_FGFR1_DEL', 'CNV_FGFR2_AMP', 'CNV_FGFR2_NORMAL',
                             'CNV_FGFR2_DEL', 'CNV_FGFR3_AMP', 'CNV_FGFR3_NORMAL', 'CNV_FGFR3_DEL', 'CNV_FGFR4_AMP',
                             'CNV_FGFR4_NORMAL', 'CNV_FGFR4_DEL', 'CNV_FLT3_AMP', 'CNV_FLT3_NORMAL', 'CNV_FLT3_DEL',
                             'CNV_IGF1R_AMP', 'CNV_IGF1R_NORMAL', 'CNV_IGF1R_DEL', 'CNV_KIT_AMP', 'CNV_KIT_NORMAL',
                             'CNV_KIT_DEL', 'CNV_KRAS_AMP', 'CNV_KRAS_NORMAL', 'CNV_KRAS_DEL', 'CNV_MDM2_AMP',
                             'CNV_MDM2_NORMAL', 'CNV_MDM2_DEL', 'CNV_MDM4_AMP', 'CNV_MDM4_NORMAL', 'CNV_MDM4_DEL',
                             'CNV_MET_AMP', 'CNV_MET_NORMAL', 'CNV_MET_DEL', 'CNV_MYC_AMP', 'CNV_MYC_NORMAL',
                             'CNV_MYC_DEL', 'CNV_MYCL_AMP', 'CNV_MYCL_NORMAL', 'CNV_MYCL_DEL', 'CNV_MYCN_AMP',
                             'CNV_MYCN_NORMAL', 'CNV_MYCN_DEL', 'CNV_NTRK1_AMP', 'CNV_NTRK1_NORMAL', 'CNV_NTRK1_DEL',
                             'CNV_NTRK2_AMP', 'CNV_NTRK2_NORMAL', 'CNV_NTRK2_DEL', 'CNV_NTRK3_AMP', 'CNV_NTRK3_NORMAL',
                             'CNV_NTRK3_DEL', 'CNV_PDGFRA_AMP', 'CNV_PDGFRA_NORMAL', 'CNV_PDGFRA_DEL', 'CNV_PDGFRB_AMP',
                             'CNV_PDGFRB_NORMAL', 'CNV_PDGFRB_DEL', 'CNV_PIK3CA_AMP', 'CNV_PIK3CA_NORMAL',
                             'CNV_PIK3CA_DEL', 'CNV_PIK3CB_AMP', 'CNV_PIK3CB_NORMAL', 'CNV_PIK3CB_DEL', 'CNV_PPARG_AMP',
                             'CNV_PPARG_NORMAL', 'CNV_PPARG_DEL', 'CNV_RICTOR_AMP', 'CNV_RICTOR_NORMAL',
                             'CNV_RICTOR_DEL', 'CNV_TERT_AMP', 'CNV_TERT_NORMAL', 'CNV_TERT_DEL', 'pN_N1', 'pN_N1A',
                             'pN_N1B', 'pN_N1C', 'pN_N1MI', 'pN_N2', 'pN_N2A', 'pN_N3', 'pN_N3A', 'pN_N3B', 'pN_N3C',
                             'pT_T1', 'pT_T1A', 'pT_T1B', 'pT_T1C', 'pT_T2', 'pT_T2B', 'pT_T3', 'pT_T4', 'pT_T4B',
                             'pT_T4D', 'IHQ_Basal', 'IHQ_Her2', 'IHQ_Luminal_A', 'IHQ_Luminal_B', 'IHQ_Normal',
                             'Tumor_IDC', 'Tumor_ILC', 'Tumor_Medullary', 'Tumor_Metaplastic', 'Tumor_Mixed',
                             'Tumor_Mucinous', 'Tumor_Other', 'Metástasis_distancia']]

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