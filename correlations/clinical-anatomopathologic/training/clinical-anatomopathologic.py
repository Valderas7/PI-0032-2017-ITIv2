import shelve # datos persistentes
import pandas as pd
import numpy as np
import imblearn
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
from tensorflow.keras.callbacks import ModelCheckpoint
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input # Para instanciar tensores de Keras
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
#filename = 'C:\\Users\\valde\Desktop\Datos_repositorio\cbioportal\data/brca_tcga_pan_can_atlas_2018.out'

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
df_all_merge.loc[df_all_merge.path_m_stage == "M0", "path_m_stage"] = 0; df_all_merge.loc[df_all_merge.path_m_stage == "M1", "path_m_stage"] = 1
df_all_merge.loc[df_all_merge.path_m_stage == "CM0 (I+)", "path_m_stage"] = 0
df_all_merge.loc[df_all_merge.os_status == "0:LIVING", "os_status"] = 0; df_all_merge.loc[df_all_merge.os_status == "1:DECEASED", "os_status"] = 1
df_all_merge.loc[df_all_merge.dfs_status == "0:DiseaseFree", "dfs_status"] = 0; df_all_merge.loc[df_all_merge.dfs_status == "1:Recurred/Progressed", "dfs_status"] = 1

""" Se crea una nueva columna para indicar la metastasis a distancia. En esta columna se indicaran los pacientes que 
tienen estadio M1 (metastasis inicial) + otros pacientes que desarrollan metastasis a lo largo de la enfermedad (para
ello se hace uso del excel pacientes_tcga y su columna DB) """
df_all_merge['distant_metastasis'] = 0
df_all_merge.loc[df_all_merge.path_m_stage == 1, 'distant_metastasis'] = 1

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
train_labels = train_tabular_data.iloc[:, 6:]
valid_labels = valid_tabular_data.iloc[:, 6:]
test_labels = test_tabular_data.iloc[:, 6:]

train_tabular_data = train_tabular_data.iloc[:, :6]
valid_tabular_data = valid_tabular_data.iloc[:, :6]
test_tabular_data = test_tabular_data.iloc[:, :6]

""" Se extraen los nombres de las columnas de las clases de salida para usarlos posteriormente en la sección de
evaluación: """
test_columns = test_labels.columns.values
classes = test_columns.tolist()

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

""" Para poder entrenar la red hace falta transformar los dataframes de entrenamiento y test en arrays de numpy, así 
como también la columna de salida de ambos subconjuntos (las imágenes YA fueron convertidas anteriormente, por lo que no
hace falta transformarlas de nuevo). """
train_tabular_data = np.asarray(train_tabular_data).astype('float32')
train_labels = np.asarray(train_labels).astype('float32')

valid_tabular_data = np.asarray(valid_tabular_data).astype('float32')
valid_labels = np.asarray(valid_labels).astype('float32')

test_tabular_data = np.asarray(test_tabular_data).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (MLP) -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
model = keras.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(train_tabular_data.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(train_labels.shape[1], activation = "sigmoid"))
model.summary()

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

model.compile(loss = 'binary_crossentropy', # Esta función de loss suele usarse para clasificación binaria.
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)

""" Se implementa un callback: para guardar el mejor modelo que tenga la menor 'loss' en la validación. """
checkpoint_path = '/home/avalderas/img_slides/correlations/clinical-anatomopathologic/inference/test_data&models/clinical-anatomopathologic.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = True, monitor= 'val_loss', mode= 'min')

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
neural_network = model.fit(x = train_tabular_data,  # Datos de entrada.
                           y = train_labels,  # Datos objetivos.
                           epochs = 200,
                           verbose = 1,
                           batch_size = 32,
                           #callbacks= mcp_save,
                           validation_data = (valid_tabular_data, valid_labels))

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_tabular_data, test_labels, verbose = 0)
print("\n'Loss' del conjunto de prueba: {:.2f}\n""Sensibilidad del conjunto de prueba: {:.2f}\n" 
      "Precisión del conjunto de prueba: {:.2f}\n""Especifidad del conjunto de prueba: {:.2f} \n"
      "Exactitud del conjunto de prueba: {:.2f} %\n" 
      "El AUC-ROC del conjunto de prueba es de: {:.2f}".format(results[0], results[5], results[6],
                                                               results[3]/(results[3]+results[2]), results[7] * 100,
                                                               results[8]))

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
print("\nGenera predicciones para 10 muestras")
print("Clase del primer paciente: \n", test_labels[:1])
np.set_printoptions(precision=3, suppress=True)
print("\nPredicciones:\n", np.round(model.predict(test_tabular_data[:1])))

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
from sklearn.metrics import multilabel_confusion_matrix
y_true = test_labels # Etiquetas verdaderas de 'test'
y_pred = np.round(model.predict(test_tabular_data)) # Predicción de etiquetas de 'test'
y_pred_prob = model.predict(test_tabular_data).ravel()

matrix = multilabel_confusion_matrix(y_true, y_pred) # Calcula (pero no dibuja) la matriz de confusión
group_names = ['True Neg','False Pos','False Neg','True Pos'] # Nombres de los grupos

conf_mat_dict = {}

""" Para cada clase, se recogen las clases verdaderas de salida y las clases predichas, para calcular la matriz de 
confusion """
for label_col in range(len(classes)):
    y_true_label = y_true[:, label_col]
    y_pred_label = y_pred[:, label_col]
    conf_mat_dict[classes[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)

for label, matrix in conf_mat_dict.items():
    print("\nMatriz de confusion para el gen {}:".format(label))
    print(matrix)
    if matrix.shape == (2,2):
        """ @zip: Une las tuplas del nombre de los grupos con la de la cantidad de casos por grupo """
        group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()]  # Cantidad de casos por grupo
        true_neg_pos_neg = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
        true_neg_pos_neg = np.asarray(true_neg_pos_neg).reshape(2, 2)
        sns.heatmap(matrix, annot=true_neg_pos_neg, fmt='', cmap='Blues')
        plt.title(label)
        plt.show()
print("\n")

""" Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases.
Para implementarlas, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva: """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
from scipy import interp

""" Para empezar, se calculan dos puntuaciones de la curva ROC. En primer lugar se calcula la puntuación micro-promedio 
(se calcula la puntuación de forma igualitaria, contando el total de todos los falsos positivos y verdaderos negativos), 
que es la mejor puntuación para ver el rendimiento de forma igualitaria del clasificador. Seguidamente, se calcula la 
puntuación promedia ponderada (se calcula la puntuación de cada una de las clases por separado, ponderando cada una de 
ellas según el número de veces que aparezca en el conjunto de datos), que es la mejor puntuación en caso de conjuntos de
datos no balanceados como el que se está analizando. """
y_pred_prob = model.predict(test_tabular_data)

micro_auc_roc = roc_auc_score(test_labels, y_pred_prob, multi_class="ovr", average="micro")
micro_auc_pr = average_precision_score(test_labels, y_pred_prob, average="micro")

weighted_auc_roc_scores = []
weighted_pr_roc_scores = []

for i in range(len(classes)):
    if len(np.unique(test_labels[:, i])) > 1:
        weighted_auc_roc_columns = roc_auc_score(test_labels[:, i], y_pred_prob[:, i], average = 'weighted')
        weighted_auc_roc_scores.append(weighted_auc_roc_columns)
        weighted_pr_roc_columns = average_precision_score(test_labels[:, i], y_pred_prob[:, i], average='weighted')
        weighted_pr_roc_scores.append(weighted_pr_roc_columns)

weighted_auc_roc = sum(weighted_auc_roc_scores) / len(weighted_auc_roc_scores)
weighted_auc_pr = sum(weighted_pr_roc_scores) / len(weighted_pr_roc_scores)

print("\nPuntuación AUC-ROC: {:.2f} (micro-promedio)".format(micro_auc_roc))
print("Puntuación AUC-PR: {:.2f} (micro-promedio)".format(micro_auc_pr))
print("\nPuntuación AUC-ROC: {:.2f} (promedio-ponderado)".format(weighted_auc_roc))
print("Puntuación AUC-PR: {:.2f} (promedio-ponderado)".format(weighted_auc_pr))

""" Una vez calculadas las dos puntuaciones, se dibuja la curva micro-promedio. Esto es mejor que dibujar una curva para 
cada una de las clases que hay en el problema. """
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(classes)):
    if len(np.unique(test_labels[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], y_pred_prob[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), y_pred_prob.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve (micro)')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(classes)):
    if len(np.unique(test_labels[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels[:, i], y_pred_prob[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels.ravel(), y_pred_prob.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve (micro)')
plt.legend(loc = 'best')
plt.show()

#np.save('test_data', test_tabular_data)
#np.save('test_labels', test_labels)