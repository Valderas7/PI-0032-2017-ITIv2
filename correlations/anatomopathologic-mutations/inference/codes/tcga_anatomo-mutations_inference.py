import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix # Para realizar la matriz de confusión

""" Se carga el modelo de red neuronal entrenado y los distintos datos de entrada y datos de salida guardados en formato 
'numpy' """
model = load_model('/correlations/anatomopathologic-mutations/inference/test_data/anatomopathologic-mutations.h5')

test_tabular_data = np.load('/correlations/anatomopathologic-mutations/inference/test_data/test_data.npy')

test_labels_snv = np.load('/correlations/anatomopathologic-mutations/inference/test_data/test_labels_snv.npy')
test_labels_cnv_a = np.load('/correlations/anatomopathologic-mutations/inference/test_data/test_labels_cnv_a.npy')
test_labels_cnv_d = np.load('/correlations/anatomopathologic-mutations/inference/test_data/test_labels_cnv_d.npy')

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
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
                 'CNV_BRAF_AMP', 'CNV_CCND1_AMP', 'CNV_CCND2_AMP', 'CNV_CCND3_AMP', 'CNV_CCNE1_AMP', 'CNV_CDK2_AMP',
                 'CNV_CDK4_AMP', 'CNV_CDK6_AMP', 'CNV_EGFR_AMP', 'CNV_ERBB2_AMP', 'CNV_ESR1_AMP', 'CNV_FGF19_AMP',
                 'CNV_FGF3_AMP', 'CNV_FGFR1_AMP', 'CNV_FGFR2_AMP', 'CNV_FGFR3_AMP', 'CNV_FGFR4_AMP', 'CNV_FLT3_AMP',
                 'CNV_IGF1R_AMP', 'CNV_KIT_AMP', 'CNV_KRAS_AMP', 'CNV_MDM2_AMP', 'CNV_MDM4_AMP', 'CNV_MET_AMP',
                 'CNV_MYC_AMP', 'CNV_MYCL_AMP', 'CNV_MYCN_AMP', 'CNV_NTRK1_AMP', 'CNV_NTRK2_AMP', 'CNV_NTRK3_AMP',
                 'CNV_PDGFRA_AMP', 'CNV_PDGFRB_AMP', 'CNV_PIK3CA_AMP', 'CNV_PIK3CB_AMP', 'CNV_PPARG_AMP',
                 'CNV_RICTOR_AMP', 'CNV_TERT_AMP']

classes_cnv_d = ['CNV_AKT1_DEL', 'CNV_AKT2_DEL', 'CNV_AKT3_DEL', 'CNV_ALK_DEL', 'CNV_AR_DEL', 'CNV_AXL_DEL',
                 'CNV_BRAF_DEL', 'CNV_CCND1_DEL', 'CNV_CCND2_DEL', 'CNV_CCND3_DEL', 'CNV_CCNE1_DEL', 'CNV_CDK2_DEL',
                 'CNV_CDK4_DEL', 'CNV_CDK6_DEL', 'CNV_EGFR_DEL', 'CNV_ERBB2_DEL', 'CNV_ESR1_DEL', 'CNV_FGF19_DEL',
                 'CNV_FGF3_DEL', 'CNV_FGFR1_DEL', 'CNV_FGFR2_DEL', 'CNV_FGFR3_DEL', 'CNV_FGFR4_DEL', 'CNV_FLT3_DEL',
                 'CNV_IGF1R_DEL', 'CNV_KIT_DEL', 'CNV_KRAS_DEL', 'CNV_MDM2_DEL', 'CNV_MDM4_DEL', 'CNV_MET_DEL',
                 'CNV_MYC_DEL', 'CNV_MYCL_DEL', 'CNV_MYCN_DEL', 'CNV_NTRK1_DEL', 'CNV_NTRK2_DEL', 'CNV_NTRK3_DEL',
                 'CNV_PDGFRA_DEL', 'CNV_PDGFRB_DEL', 'CNV_PIK3CA_DEL', 'CNV_PIK3CB_DEL', 'CNV_PPARG_DEL',
                 'CNV_RICTOR_DEL', 'CNV_TERT_DEL']

index_snv = 0

for matrix_gen_snv in matrix_snv:
    group_counts = ['{0:0.0f}'.format(value) for value in matrix_gen_snv.flatten()]  # Cantidad de casos por grupo
    true_neg_pos_neg = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
    true_neg_pos_neg = np.asarray(true_neg_pos_neg).reshape(2, 2)
    sns.heatmap(matrix_gen_snv, annot=true_neg_pos_neg, fmt='', cmap='Blues')
    plt.title('Mutación SNV del gen {}'.format(classes_snv[index_snv].split('_')[1]))
    #plt.savefig('/home/avalderas/img_slides/screenshots/correlations/anatomopathologic - mutations/tcga/SNV/{}'.format(classes_snv[index_snv].split('_')[1]), bbox_inches='tight')
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
    #plt.savefig('/home/avalderas/img_slides/screenshots/correlations/anatomopathologic - mutations/tcga/CNV-A/{}'.format(classes_cnv_a[index_cnv_a].split('_')[1]), bbox_inches='tight')
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
    #plt.savefig('/home/avalderas/img_slides/screenshots/correlations/anatomopathologic - mutations/tcga/CNV-D/{}'.format(classes_cnv_d[index_cnv_d].split('_')[1]), bbox_inches='tight')
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