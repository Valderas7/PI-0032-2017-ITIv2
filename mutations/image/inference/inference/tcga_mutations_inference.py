""" Programa para evaluar las métricas del subconjunto de test (conjunto total de imágenes de TCGA de las mutaciones
SNV, CNV-A y CNV-D).
Además, se realizan las curvas AUC-ROC y AUC-PR de los genes más repetidos en el estudio que ha
realizado Irene con los pacientes del proyecto de INiBICA:

Table 2. Mutations samples in TCGA dataset with lymph node patients (552 patients) by gen

|       SNVs        |       CNVs-A         |      CNVs-D         |
|  Gene  | Samples ||   Gene  |   Samples ||    Gene  | Samples |
PIK3CA      171         MYC         81          BRCA2       5
TP53        168         CCND1       84          BRCA1       3
AKT1        13          CDKN1B      1           KDR         1
PTEN        31          FGF19       80          CHEK1       5
ERBB2       12          ERBB2       79          FGF3        0
EGFR        4           FGF3        81          FANCA       12
MTOR        9
"""

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
from sklearn.metrics import confusion_matrix

""" Se carga el modelo de red neuronal entrenado y los distintos datos de entrada y datos de salida guardados en formato 
'numpy' """
model = load_model('/home/avalderas/img_slides/mutations/image/inference/models/model_image_mutations_25_0.41.h5')

test_image_data = np.load('/home/avalderas/img_slides/mutations/image/inference/test_data/test_image.npy')

test_labels_snv = np.load('/home/avalderas/img_slides/mutations/image/inference/test_data/test_labels_snv.npy')
test_labels_cnv_a = np.load('/home/avalderas/img_slides/mutations/image/inference/test_data/test_labels_cnv_a.npy')
test_labels_cnv_normal = np.load('/home/avalderas/img_slides/mutations/image/inference/test_data/test_labels_cnv_normal.npy')
test_labels_cnv_d = np.load('/home/avalderas/img_slides/mutations/image/inference/test_data/test_labels_cnv_d.npy')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data, [test_labels_snv, test_labels_cnv_a, test_labels_cnv_normal,
                                           test_labels_cnv_d], verbose = 0)
"""
print("\n'Loss' de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "SNV del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones SNV del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f} \n"
      "Exactitud de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones SNV"
      " del panel OCA en el conjunto de prueba: {:.2f}\nAUC-PR de las mutaciones SNV del panel OCA en el conjunto de "
      "prueba: {:.2f}".format(results[1], results[9], results[10], results[7]/(results[7]+results[6]),
                              results[11] * 100, results[12], results[13]))

if results[9] > 0 or results[10] > 0:
    print("Valor-F de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[9] * results[10]) /
                                                                                                (results[9] + results[10])))

print("\n'Loss' de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones CNV-A del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n"
      "Exactitud de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones "
      "CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""AUC-PR de las mutaciones CNV-A del panel OCA en el "
      "conjunto de prueba: {:.2f}".format(results[2], results[18], results[19], results[16]/(results[16]+results[15]),
                                          results[20] * 100, results[21], results[22]))

if results[18] > 0 or results[19] > 0:
    print("Valor-F de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[18] * results[19]) /
                                                                                                  (results[18] + results[19])))

print("\n'Loss' de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones CNV-D del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n"
      "Exactitud de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones "
      "CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""AUC-PR de las mutaciones CNV-D del panel OCA en el "
      "conjunto de prueba: {:.2f}".format(results[4], results[36], results[37], results[34]/(results[34]+results[33]),
                                          results[38] * 100, results[39], results[40]))
if results[36] > 0 or results[37] > 0:
    print("Valor-F de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[36] * results[37]) /
                                                                                                  (results[36] + results[37])))
"""
""" -------------------------------------------------------------------------------------------------------------------
------------------------------------------- SECCIÓN DE EVALUACIÓN  ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Se definen las clases de salida de los genes SNV, CNV-A y CNV-D """
classes_snv = ['SNV_AKT1', 'SNV_AKT2', 'SNV_AKT3', 'SNV_ALK', 'SNV_AR', 'SNV_ARAF', 'SNV_AXL', 'SNV_BRAF', 'SNV_BTK',
               'SNV_CBL', 'SNV_CCND1', 'SNV_CDK4', 'SNV_CDK6', 'SNV_CHEK2', 'SNV_CSF1R', 'SNV_CTNNB1', 'SNV_DDR2',
               'SNV_EGFR', 'SNV_ERBB2', 'SNV_ERBB3', 'SNV_ERBB4', 'SNV_ERCC2', 'SNV_ESR1', 'SNV_EZH2', 'SNV_FGFR1',
               'SNV_FGFR2', 'SNV_FGFR3', 'SNV_FGFR4', 'SNV_FLT3', 'SNV_FOXL2', 'SNV_GATA2', 'SNV_GNA11', 'SNV_GNAQ',
               'SNV_GNAS', 'SNV_H3F3A', 'SNV_HIST1H3B', 'SNV_HNF1A', 'SNV_HRAS', 'SNV_IDH1', 'SNV_IDH2', 'SNV_JAK1',
               'SNV_JAK2', 'SNV_JAK3', 'SNV_KDR', 'SNV_KIT', 'SNV_KNSTRN', 'SNV_KRAS', 'SNV_MAGOH', 'SNV_MAP2K1',
               'SNV_MAP2K2', 'SNV_MAP2K4', 'SNV_MAPK1', 'SNV_MAX', 'SNV_MDM4', 'SNV_MED12', 'SNV_MET', 'SNV_MTOR',
               'SNV_MYC', 'SNV_MYCN', 'SNV_MYD88', 'SNV_NFE2L2', 'SNV_NRAS', 'SNV_NTRK1', 'SNV_NTRK2', 'SNV_NTRK3',
               'SNV_PDGFRA', 'SNV_PDGFRB', 'SNV_PIK3CA', 'SNV_PIK3CB', 'SNV_PPP2R1A', 'SNV_PTPN11', 'SNV_RAC1',
               'SNV_RAF1', 'SNV_RET', 'SNV_RHEB', 'SNV_RHOA', 'SNV_ROS1', 'SNV_SF3B1', 'SNV_SMAD4', 'SNV_SMO',
               'SNV_SPOP', 'SNV_SRC', 'SNV_STAT3', 'SNV_TERT', 'SNV_TOP1', 'SNV_U2AF1', 'SNV_XPO1', 'SNV_BRCA1',
               'SNV_BRCA2', 'SNV_CDKN2A', 'SNV_ERG', 'SNV_ETV1', 'SNV_ETV4', 'SNV_ETV5', 'SNV_FGR', 'SNV_MYB',
               'SNV_MYBL1', 'SNV_NF1', 'SNV_NOTCH1', 'SNV_NOTCH4', 'SNV_NRG1', 'SNV_NUTM1', 'SNV_PPARG', 'SNV_PRKACA',
               'SNV_PRKACB', 'SNV_PTEN', 'SNV_RAD51B', 'SNV_RB1', 'SNV_RELA', 'SNV_RSPO2', 'SNV_RSPO3', 'SNV_ARID1A',
               'SNV_ATM', 'SNV_ATR', 'SNV_ATRX', 'SNV_BAP1', 'SNV_CDK12', 'SNV_CDKN1B', 'SNV_CDKN2B', 'SNV_CHEK1',
               'SNV_CREBBP', 'SNV_FANCA', 'SNV_FANCD2', 'SNV_FANCI', 'SNV_FBXW7', 'SNV_MLH1', 'SNV_MRE11', 'SNV_MSH2',
               'SNV_MSH6', 'SNV_NBN', 'SNV_NF2', 'SNV_NOTCH2', 'SNV_NOTCH3', 'SNV_PALB2', 'SNV_PIK3R1', 'SNV_PMS2',
               'SNV_POLE', 'SNV_PTCH1', 'SNV_RAD50', 'SNV_RAD51', 'SNV_RAD51C', 'SNV_RAD51D', 'SNV_RNF43', 'SNV_SETD2',
               'SNV_SLX4', 'SNV_SMARCA4', 'SNV_SMARCB1', 'SNV_STK11', 'SNV_TP53', 'SNV_TSC1', 'SNV_TSC2']

classes_cnv_a = ['CNV_AKT1_AMP', 'CNV_AKT2_AMP', 'CNV_AKT3_AMP', 'CNV_ALK_AMP', 'CNV_AR_AMP', 'CNV_AXL_AMP', 
                 'CNV_BRCA1_AMP', 'CNV_BRCA2_AMP', 'CNV_BRAF_AMP', 'CNV_CCND1_AMP', 'CNV_CCND2_AMP', 'CNV_CCND3_AMP', 
                 'CNV_CCNE1_AMP', 'CNV_CDK2_AMP', 'CNV_CDK4_AMP', 'CNV_CDK6_AMP', 'CNV_CDKN1B_AMP', 'CNV_CHEK1_AMP',
                 'CNV_EGFR_AMP', 'CNV_ERBB2_AMP', 'CNV_ESR1_AMP', 'CNV_FANCA_AMP', 'CNV_FGF19_AMP', 'CNV_FGF3_AMP',
                 'CNV_FGFR1_AMP', 'CNV_FGFR2_AMP', 'CNV_FGFR3_AMP', 'CNV_FGFR4_AMP', 'CNV_FLT3_AMP', 'CNV_IGF1R_AMP',
                 'CNV_KDR_AMP', 'CNV_KIT_AMP', 'CNV_KRAS_AMP', 'CNV_MDM2_AMP', 'CNV_MDM4_AMP', 'CNV_MET_AMP',
                 'CNV_MYC_AMP', 'CNV_MYCL_AMP', 'CNV_MYCN_AMP', 'CNV_NTRK1_AMP', 'CNV_NTRK2_AMP', 'CNV_NTRK3_AMP',
                 'CNV_PDGFRA_AMP', 'CNV_PDGFRB_AMP', 'CNV_PIK3CA_AMP', 'CNV_PIK3CB_AMP', 'CNV_PPARG_AMP',
                 'CNV_RICTOR_AMP', 'CNV_TERT_AMP']

classes_cnv_d = ['CNV_AKT1_DEL', 'CNV_AKT2_DEL', 'CNV_AKT3_DEL', 'CNV_ALK_DEL', 'CNV_AR_DEL', 'CNV_AXL_DEL',
                 'CNV_BRCA1_DEL', 'CNV_BRCA2_DEL', 'CNV_BRAF_DEL', 'CNV_CCND1_DEL', 'CNV_CCND2_DEL', 'CNV_CCND3_DEL',
                 'CNV_CCNE1_DEL', 'CNV_CDK2_DEL', 'CNV_CDK4_DEL', 'CNV_CDK6_DEL', 'CNV_CDKN1B_DEL', 'CNV_CHEK1_DEL',
                 'CNV_EGFR_DEL', 'CNV_ERBB2_DEL', 'CNV_ESR1_DEL', 'CNV_FANCA_DEL', 'CNV_FGF19_DEL', 'CNV_FGF3_DEL', 
                 'CNV_FGFR1_DEL', 'CNV_FGFR2_DEL', 'CNV_FGFR3_DEL', 'CNV_FGFR4_DEL', 'CNV_FLT3_DEL', 'CNV_IGF1R_DEL', 
                 'CNV_KDR_DEL', 'CNV_KIT_DEL', 'CNV_KRAS_DEL', 'CNV_MDM2_DEL', 'CNV_MDM4_DEL', 'CNV_MET_DEL',
                 'CNV_MYC_DEL', 'CNV_MYCL_DEL', 'CNV_MYCN_DEL', 'CNV_NTRK1_DEL', 'CNV_NTRK2_DEL', 'CNV_NTRK3_DEL',
                 'CNV_PDGFRA_DEL', 'CNV_PDGFRB_DEL', 'CNV_PIK3CA_DEL', 'CNV_PIK3CB_DEL', 'CNV_PPARG_DEL',
                 'CNV_RICTOR_DEL', 'CNV_TERT_DEL']


""" Códigos para calcular las curvas AUC-ROC y AUC-PR micro de todos los subconjuntos de mutaciones para todos los genes

#Para finalizar, se dibuja el área bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
#documento gráfico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
#tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases:
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interp

y_pred_prob_snv = model.predict(test_image_data)[0]
y_pred_prob_cnv_a = model.predict(test_image_data)[1]
y_pred_prob_cnv_d = model.predict(test_image_data)[3]

# SNV
fpr = dict()
tpr = dict()
auc_roc = dict()

#Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
#de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio 
for i in range(len(classes_snv)):
    if len(np.unique(test_labels_snv[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_snv[:, i], y_pred_prob_snv[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_snv.ravel(), y_pred_prob_snv.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

#Finalmente se dibuja la curva AUC-ROC micro-promedio
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for SNV mutations')
plt.legend(loc = 'best')
plt.show()

#Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
#del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados.
precision = dict()
recall = dict()
auc_pr = dict()

#Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
#clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio
for i in range(len(classes_snv)):
    if len(np.unique(test_labels_snv[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_snv[:, i], y_pred_prob_snv[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_snv.ravel(), y_pred_prob_snv.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

#Finalmente se dibuja la curvas AUC-PR micro-promedio
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve (micro) for SNV mutations')
plt.legend(loc = 'best')
plt.show()

# CNV-A
fpr = dict()
tpr = dict()
auc_roc = dict()

#Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
#de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio
for i in range(len(classes_cnv_a)):
    if len(np.unique(test_labels_cnv_a[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_cnv_a[:, i], y_pred_prob_cnv_a[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_cnv_a.ravel(), y_pred_prob_cnv_a.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

#Finalmente se dibuja la curva AUC-ROC micro-promedio
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for CNV-A mutations')
plt.legend(loc = 'best')
plt.show()

#Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
#del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados.
precision = dict()
recall = dict()
auc_pr = dict()

#Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
#clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio
for i in range(len(classes_cnv_a)):
    if len(np.unique(test_labels_cnv_a[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_cnv_a[:, i], y_pred_prob_cnv_a[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_cnv_a.ravel(), y_pred_prob_cnv_a.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

#Finalmente se dibuja la curvas AUC-PR micro-promedio
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for CNV-A mutations')
plt.legend(loc = 'best')
plt.show()

#CNV-D
fpr = dict()
tpr = dict()
auc_roc = dict()

#Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
#de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio
for i in range(len(classes_cnv_d)):
    if len(np.unique(test_labels_cnv_d[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_cnv_d[:, i], y_pred_prob_cnv_d[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_cnv_d.ravel(), y_pred_prob_cnv_d.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

#Finalmente se dibuja la curva AUC-ROC micro-promedio
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for CNV-D mutations')
plt.legend(loc = 'best')
plt.show()

#Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
#del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados.
precision = dict()
recall = dict()
auc_pr = dict()

#Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
#clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio
for i in range(len(classes_cnv_d)):
    if len(np.unique(test_labels_cnv_d[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_cnv_d[:, i], y_pred_prob_cnv_d[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_cnv_d.ravel(), y_pred_prob_cnv_d.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

#Finalmente se dibuja la curvas AUC-PR micro-promedio
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for CNV-D mutations')
plt.legend(loc = 'best')
plt.show()
"""

""" Para finalizar, se dibuja el área bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento gráfico del rendimiento del clasificador binario de los genes sobre los que se van a realizar posteriormente
los mapas de calor. Esta curva representa la tasa de verdaderos positivos y la tasa de falsos positivos, por lo que 
resume el comportamiento general del clasificador para diferenciar clases. También se dibuja el área bajo la la curva PR 
(precision-recall), para tener un documento gráfico del rendimiento del clasificador en cuanto a la sensibilidad y la 
precisión de resultados.
 
Para terminar, se calculan las métricas específicas (sensibilidad, precisión, eficacia y especificidad) de estos genes
sobre los que se van a realizar los mapas de calor. """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report

""" --------------------------------------------------------------------------------------------------------------------
------------------------------------------------------ SNV -------------------------------------------------------------  
------------------------------------------------------------------------------------------------------------------------ """
""" ------------------------------------------------- PIK3CA ----------------------------------------------------------- """
y_true = test_labels_snv[:, classes_snv.index('SNV_PIK3CA')]
y_pred_prob = model.predict(test_image_data)[0]
y_pred_prob = y_pred_prob[:, classes_snv.index('SNV_PIK3CA')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_snv_pik3ca = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for SNV of PIK3CA gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_snv_pik3ca = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for SNV of PIK3CA gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones SNV del gen PIK3CA: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones SNV del gen PIK3CA: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones SNV del gen PIK3CA: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones SNV del gen PIK3CA: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones SNV del gen PIK3CA: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones SNV del gen PIK3CA: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones SNV del gen PIK3CA: {:.2f}%".format(accuracy * 100))

""" -------------------------------------------------- TP53 ------------------------------------------------------------ """
y_true = test_labels_snv[:, classes_snv.index('SNV_TP53')]
y_pred_prob = model.predict(test_image_data)[0]
y_pred_prob = y_pred_prob[:, classes_snv.index('SNV_TP53')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_snv_tp53 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for SNV of TP53 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_snv_tp53 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for SNV of TP53 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones SNV del gen TP53: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones SNV del gen TP53: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones SNV del gen TP53: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones SNV del gen TP53: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones SNV del gen TP53: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones SNV del gen TP53: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones SNV del gen TP53: {:.2f}%".format(accuracy * 100))

""" -------------------------------------------------- AKT1 ------------------------------------------------------------ """
y_true = test_labels_snv[:, classes_snv.index('SNV_AKT1')]
y_pred_prob = model.predict(test_image_data)[0]
y_pred_prob = y_pred_prob[:, classes_snv.index('SNV_AKT1')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_snv_akt1 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for SNV of AKT1 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_snv_akt1 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for SNV of AKT1 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones SNV del gen AKT1: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones SNV del gen AKT1: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones SNV del gen AKT1: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones SNV del gen AKT1: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones SNV del gen AKT1: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones SNV del gen AKT1: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones SNV del gen AKT1: {:.2f}%".format(accuracy * 100))

""" -------------------------------------------------- PTEN ------------------------------------------------------------ """
y_true = test_labels_snv[:, classes_snv.index('SNV_PTEN')]
y_pred_prob = model.predict(test_image_data)[0]
y_pred_prob = y_pred_prob[:, classes_snv.index('SNV_PTEN')].ravel()
if 1 in y_true:
    # AUC-ROC
    fpr, tpr, thresholds_snv_pten = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for SNV of PTEN gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_snv_pten = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for SNV of PTEN gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones SNV del gen PTEN: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones SNV del gen PTEN: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones SNV del gen PTEN: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones SNV del gen PTEN: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones SNV del gen PTEN: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones SNV del gen PTEN: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones SNV del gen PTEN: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- ERBB2 ---------------------------------------------------------- """
y_true = test_labels_snv[:, classes_snv.index('SNV_ERBB2')]
y_pred_prob = model.predict(test_image_data)[0]
y_pred_prob = y_pred_prob[:, classes_snv.index('SNV_ERBB2')].ravel()
if 1 in y_true:
    # AUC-ROC
    fpr, tpr, thresholds_snv_erbb2 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for SNV of ERBB2 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_snv_erbb2 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for SNV of ERBB2 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones SNV del gen ERBB2: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones SNV del gen ERBB2: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones SNV del gen ERBB2: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones SNV del gen ERBB2: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones SNV del gen ERBB2: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones SNV del gen ERBB2: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones SNV del gen ERBB2: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- EGFR ---------------------------------------------------------- """
y_true = test_labels_snv[:, classes_snv.index('SNV_EGFR')]
y_pred_prob = model.predict(test_image_data)[0]
y_pred_prob = y_pred_prob[:, classes_snv.index('SNV_EGFR')].ravel()
if 1 in y_true:
    # AUC-ROC
    fpr, tpr, thresholds_snv_egfr = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for SNV of EGFR gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_snv_egfr = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for SNV of EGFR gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones SNV del gen EGFR: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones SNV del gen EGFR: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones SNV del gen EGFR: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones SNV del gen EGFR: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones SNV del gen EGFR: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones SNV del gen EGFR: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones SNV del gen EGFR: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- MTOR ---------------------------------------------------------- """
y_true = test_labels_snv[:, classes_snv.index('SNV_MTOR')]
y_pred_prob = model.predict(test_image_data)[0]
y_pred_prob = y_pred_prob[:, classes_snv.index('SNV_MTOR')].ravel()
if 1 in y_true:
    # AUC-ROC
    fpr, tpr, thresholds_snv_mtor = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for SNV of MTOR gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_snv_mtor = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for SNV of MTOR gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones SNV del gen MTOR: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones SNV del gen MTOR: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones SNV del gen MTOR: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones SNV del gen MTOR: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones SNV del gen MTOR: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones SNV del gen MTOR: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones SNV del gen MTOR: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------------------------------------------------------------------------
------------------------------------------------------ CNV-A -----------------------------------------------------------  
------------------------------------------------------------------------------------------------------------------------ """
""" --------------------------------------------------- MYC ---------------------------------------------------------- """
y_true = test_labels_cnv_a[:, classes_cnv_a.index('CNV_MYC_AMP')]
y_pred_prob = model.predict(test_image_data)[1]
y_pred_prob = y_pred_prob[:, classes_cnv_a.index('CNV_MYC_AMP')].ravel()
if 1 in y_true:
    # AUC-ROC
    fpr, tpr, thresholds_cnv_a_myc = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-A of MYC gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_a_myc = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-A of MYC gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-A del gen MYC: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-A del gen MYC: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-A del gen MYC: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-A del gen MYC: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-A del gen MYC: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-A del gen MYC: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-A del gen MYC: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- CCND1 ---------------------------------------------------------- """
y_true = test_labels_cnv_a[:, classes_cnv_a.index('CNV_CCND1_AMP')]
y_pred_prob = model.predict(test_image_data)[1]
y_pred_prob = y_pred_prob[:, classes_cnv_a.index('CNV_CCND1_AMP')].ravel()
if 1 in y_true:
    # AUC-ROC
    fpr, tpr, thresholds_cnv_a_ccnd1 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-A of CCND1 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_a_ccnd1 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-A of CCND1 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-A del gen CCND1: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-A del gen CCND1: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-A del gen CCND1: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-A del gen CCND1: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-A del gen CCND1: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-A del gen CCND1: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-A del gen CCND1: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- CDKN1B --------------------------------------------------------- """
y_true = test_labels_cnv_a[:, classes_cnv_a.index('CNV_CDKN1B_AMP')]
y_pred_prob = model.predict(test_image_data)[1]
y_pred_prob = y_pred_prob[:, classes_cnv_a.index('CNV_CDKN1B_AMP')].ravel()
if 1 in y_true:
    # AUC-ROC
    fpr, tpr, thresholds_cnv_a_cdkn1b = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-A of CDKN1B gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_a_cdkn1b = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-A of CDKN1B gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-A del gen CDKN1B: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-A del gen CDKN1B: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-A del gen CDKN1B: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-A del gen CDKN1B: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-A del gen CDKN1B: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-A del gen CDKN1B: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-A del gen CDKN1B: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- FGF19 ---------------------------------------------------------- """
y_true = test_labels_cnv_a[:, classes_cnv_a.index('CNV_FGF19_AMP')]
y_pred_prob = model.predict(test_image_data)[1]
y_pred_prob = y_pred_prob[:, classes_cnv_a.index('CNV_FGF19_AMP')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_cnv_a_fgf19 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-A of FGF19 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_a_fgf19 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-A of FGF19 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-A del gen FGF19: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-A del gen FGF19: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-A del gen FGF19: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-A del gen FGF19: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-A del gen FGF19: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-A del gen FGF19: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-A del gen FGF19: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- ERBB2 ---------------------------------------------------------- """
y_true = test_labels_cnv_a[:, classes_cnv_a.index('CNV_ERBB2_AMP')]
y_pred_prob = model.predict(test_image_data)[1]
y_pred_prob = y_pred_prob[:, classes_cnv_a.index('CNV_ERBB2_AMP')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_cnv_a_erbb2 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-A of ERBB2 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_a_erbb2 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-A of ERBB2 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-A del gen ERBB2: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-A del gen ERBB2: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-A del gen ERBB2: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-A del gen ERBB2: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-A del gen ERBB2: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-A del gen ERBB2: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-A del gen ERBB2: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- FGF3 ---------------------------------------------------------- """
y_true = test_labels_cnv_a[:, classes_cnv_a.index('CNV_FGF3_AMP')]
y_pred_prob = model.predict(test_image_data)[1]
y_pred_prob = y_pred_prob[:, classes_cnv_a.index('CNV_FGF3_AMP')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_cnv_a_fgf3 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-A of FGF3 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_a_fgf3 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-A of FGF3 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-A del gen FGF3: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-A del gen FGF3: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-A del gen FGF3: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-A del gen FGF3: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-A del gen FGF3: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-A del gen FGF3: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-A del gen FGF3: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------------------------------------------------------------------------
------------------------------------------------------- CNV-D ----------------------------------------------------------  
------------------------------------------------------------------------------------------------------------------------ """
""" --------------------------------------------------- BRCA1 ---------------------------------------------------------- """
y_true = test_labels_cnv_d[:, classes_cnv_d.index('CNV_BRCA1_DEL')]
y_pred_prob = model.predict(test_image_data)[3]
y_pred_prob = y_pred_prob[:, classes_cnv_d.index('CNV_BRCA1_DEL')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_cnv_d_brca1 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-D of BRCA1 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_d_brca1 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-D of BRCA1 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-D del gen BRCA1: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-D del gen BRCA1: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-D del gen BRCA1: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-D del gen BRCA1: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-D del gen BRCA1: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-D del gen BRCA1: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-D del gen BRCA1: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- BRCA2 ---------------------------------------------------------- """
y_true = test_labels_cnv_d[:, classes_cnv_d.index('CNV_BRCA2_DEL')]
y_pred_prob = model.predict(test_image_data)[3]
y_pred_prob = y_pred_prob[:, classes_cnv_d.index('CNV_BRCA2_DEL')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_cnv_d_brca2 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-D of BRCA2 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_d_brca2 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-D of BRCA2 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-D del gen BRCA2: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-D del gen BRCA2: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-D del gen BRCA2: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-D del gen BRCA2: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-D del gen BRCA2: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-D del gen BRCA2: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-D del gen BRCA2: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- KDR ------------------------------------------------------------ """
y_true = test_labels_cnv_d[:, classes_cnv_d.index('CNV_KDR_DEL')]
y_pred_prob = model.predict(test_image_data)[3]
y_pred_prob = y_pred_prob[:, classes_cnv_d.index('CNV_KDR_DEL')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_cnv_d_kdr = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-D of KDR gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_d_kdr = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-D of KDR gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-D del gen KDR: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-D del gen KDR: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-D del gen KDR: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-D del gen KDR: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-D del gen KDR: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-D del gen KDR: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-D del gen KDR: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- CHEK1 ---------------------------------------------------------- """
y_true = test_labels_cnv_d[:, classes_cnv_d.index('CNV_CHEK1_DEL')]
y_pred_prob = model.predict(test_image_data)[3]
y_pred_prob = y_pred_prob[:, classes_cnv_d.index('CNV_CHEK1_DEL')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_cnv_d_chek1 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-D of CHEK1 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_d_chek1 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-D of CHEK1 gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-D del gen CHEK1: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-D del gen CHEK1: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-D del gen CHEK1: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-D del gen CHEK1: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-D del gen CHEK1: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-D del gen CHEK1: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-D del gen CHEK1: {:.2f}%".format(accuracy * 100))

""" --------------------------------------------------- FGF3 ---------------------------------------------------------- """
y_true = test_labels_cnv_d[:, classes_cnv_d.index('CNV_FGF3_DEL')]
y_pred_prob = model.predict(test_image_data)[3]
y_pred_prob = y_pred_prob[:, classes_cnv_d.index('CNV_FGF3_DEL')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_cnv_d_fgf3 = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-D of FGF3 gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_d_fgf3 = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-D of FGF3 gene')
    plt.legend(loc = 'best')
    plt.show()
    
    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-D del gen FGF3: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-D del gen FGF3: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-D del gen FGF3: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-D del gen FGF3: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-D del gen FGF3: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-D del gen FGF3: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-D del gen FGF3: {:.2f}%".format(accuracy * 100))

""" -------------------------------------------------- FANCA ----------------------------------------------------------- """
y_true = test_labels_cnv_d[:, classes_cnv_d.index('CNV_FANCA_DEL')]
y_pred_prob = model.predict(test_image_data)[3]
y_pred_prob = y_pred_prob[:, classes_cnv_d.index('CNV_FANCA_DEL')].ravel()

if (y_true == 1).any():
    # AUC-ROC
    fpr, tpr, thresholds_cnv_d_fanca = roc_curve(y_true, y_pred_prob)
    auc_roc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for CNV-D of FANCA gene')
    plt.legend(loc = 'best')
    plt.show()

    # AUC-PR
    precision, recall, threshold_cnv_d_fanca = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = auc(recall, precision)
    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for CNV-D of FANCA gene')
    plt.legend(loc = 'best')
    plt.show()

    # Métricas
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_pred_prob)).ravel()
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print("\nSensibilidad de las mutaciones CNV-D del gen FANCA: {:.2f}%".format(recall * 100))
    else:
        recall = "No definido"
        print("\nSensibilidad de las mutaciones CNV-D del gen FANCA: {}".format(recall))
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print("Precisión de las mutaciones CNV-D del gen FANCA: {:.2f}%".format(precision * 100))
    else:
        precision = "No definido"
        print("Precisión de las mutaciones CNV-D del gen FANCA: {}".format(precision))
    if tn + fp > 0:
        specifity = tn / (tn + fp)
        print("Especifidad de las mutaciones CNV-D del gen FANCA: {:.2f}%".format(specifity * 100))
    else:
        specifity = "No definido"
        print("Especifidad de las mutaciones CNV-D del gen FANCA: {}".format(specifity))
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print("Exactitud de las mutaciones CNV-D del gen FANCA: {:.2f}%".format(accuracy * 100))