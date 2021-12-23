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
from sklearn.metrics import confusion_matrix # Para realizar la matriz de confusión

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN DATOS TABULARES ------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# 1) Datos de anatomía patológica: Tipo histológico, STAGE, pT, pN, pM, IHQ.
# 2) Datos clínicos: Metástasis a distancia, supervivencia, recaída.
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

df_list = [df_os_status, df_dfs_status, df_tumor_type, df_path_m_stage, df_path_n_stage, df_path_t_stage, df_stage,
           df_subtype]

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

""" Se reordenan las columnas del dataframe """
cols = df_all_merge.columns.tolist()
cols = cols[0:1] + cols[4:] + cols[1:4]
df_all_merge = df_all_merge[cols]

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
# 299train, 34val, 84test
train_tabular_data, test_tabular_data = train_test_split(df_all_merge, test_size = 0.20)
train_tabular_data, valid_tabular_data = train_test_split(train_tabular_data, test_size = 0.10)

""" Ya se puede eliminar de los dos subconjuntos la columna 'ID' que no es útil para la red MLP: """
train_tabular_data = train_tabular_data.drop(['ID'], axis=1)
valid_tabular_data = valid_tabular_data.drop(['ID'], axis=1)
test_tabular_data = test_tabular_data.drop(['ID'], axis=1)

""" Se dividen los datos de anatomía patológica y los datos clínicos en datos de entrada y datos de salida."""
train_labels_metastasis = train_tabular_data.iloc[:, -1]
valid_labels_metastasis = valid_tabular_data.iloc[:, -1]
test_labels_metastasis = test_tabular_data.iloc[:, -1]

train_labels_survival = train_tabular_data.iloc[:, -3]
valid_labels_survival = valid_tabular_data.iloc[:, -3]
test_labels_survival = test_tabular_data.iloc[:, -3]

train_labels_relapse = train_tabular_data.iloc[:, -2]
valid_labels_relapse = valid_tabular_data.iloc[:, -2]
test_labels_relapse = test_tabular_data.iloc[:, -2]

train_tabular_data = train_tabular_data.iloc[:, 0:-3]
valid_tabular_data = valid_tabular_data.iloc[:, 0:-3]
test_tabular_data = test_tabular_data.iloc[:, 0:-3]

""" Para poder entrenar la red hace falta transformar los dataframes de entrenamiento y test en arrays de numpy. """
train_tabular_data = np.asarray(train_tabular_data).astype('float32')
train_labels_metastasis = np.asarray(train_labels_metastasis).astype('float32')
train_labels_survival = np.asarray(train_labels_survival).astype('float32')
train_labels_relapse = np.asarray(train_labels_relapse).astype('float32')

valid_tabular_data = np.asarray(valid_tabular_data).astype('float32')
valid_labels_metastasis = np.asarray(valid_labels_metastasis).astype('float32')
valid_labels_survival = np.asarray(valid_labels_survival).astype('float32')
valid_labels_relapse = np.asarray(valid_labels_relapse).astype('float32')

test_tabular_data = np.asarray(test_tabular_data).astype('float32')
test_labels_metastasis = np.asarray(test_labels_metastasis).astype('float32')
test_labels_survival = np.asarray(test_labels_survival).astype('float32')
test_labels_relapse = np.asarray(test_labels_relapse).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (MLP) -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
Input_ = Input(shape=train_tabular_data.shape[1], )
model = layers.Dense(128, activation = 'relu')(Input_)
model = layers.Dropout(0.5)(model)
model = layers.Dense(64, activation = 'relu')(model)
model = layers.Dropout(0.5)(model)
output1 = layers.Dense(1, activation ="sigmoid", name='metastasis')(model)
output2 = layers.Dense(1, activation ="sigmoid", name ='survival')(model)
output3 = layers.Dense(1, activation ="sigmoid", name='relapse')(model)

model = Model(inputs = Input_, outputs = [output1, output2, output3])

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

model.compile(loss = {'metastasis': 'binary_crossentropy', 'survival': 'binary_crossentropy',
                      'relapse': 'binary_crossentropy'},
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)

model.summary()

""" Se implementa un callback: para guardar el mejor modelo que tenga la menor 'loss' en la validación. """
checkpoint_path = '/correlations/anatomopathologic-clinical/inference/models/anatomopathologic-clinical.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = True, monitor= 'loss', mode= 'min')

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
neural_network = model.fit(x = train_tabular_data,
                           y = {'metastasis': train_labels_metastasis, 'survival': train_labels_survival,
                                'relapse': train_labels_relapse},
                           epochs = 1000,
                           verbose = 1,
                           batch_size = 32,
                           #callbacks= mcp_save,
                           validation_data = (valid_tabular_data, {'metastasis': valid_labels_metastasis,
                                                                   'survival': valid_labels_survival,
                                                                   'relapse': valid_labels_relapse}))

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
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
conjunto de datos de test que se definió anteriormente al repartir los datos. 
Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
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

# Metástasis a distancia
y_true_metastasis = test_labels_metastasis
y_pred_metastasis = np.round(model.predict(test_tabular_data)[0])

matrix_metastasis = confusion_matrix(y_true_metastasis, y_pred_metastasis, labels = [0, 1])
matrix_metastasis_classes = ['No Distant Metastasis', 'Distant Metastasis']

plot_confusion_matrix(matrix_metastasis, classes = matrix_metastasis_classes, title = 'Matriz de confusión de metástasis'
                                                                                      ' a distancia')
plt.show()

# Supervivencia
y_true_survival = test_labels_survival
y_pred_survival = np.round(model.predict(test_tabular_data)[1])

matrix_survival = confusion_matrix(y_true_survival, y_pred_survival, labels = [0, 1])
matrix_survival_classes = ['Living', 'Deceased']

plot_confusion_matrix(matrix_survival, classes = matrix_survival_classes, title = 'Matriz de confusión de supervivencia'
                                                                                  ' del paciente')
plt.show()

# Recidivas
y_true_relapse = test_labels_relapse
y_pred_relapse = np.round(model.predict(test_tabular_data)[2])

matrix_relapse = confusion_matrix(y_true_relapse, y_pred_relapse, labels = [0, 1])
matrix_relapse_classes = ['Disease-Free', 'Recurred']

plot_confusion_matrix(matrix_relapse, classes = matrix_relapse_classes, title = 'Matriz de confusión de recidivas')
plt.show()

#np.save('test_data', test_tabular_data)
#np.save('test_labels_metastasis', test_labels_metastasis)
#np.save('test_labels_survival', test_labels_survival)
#np.save('test_labels_metastasis', test_labels_metastasis)