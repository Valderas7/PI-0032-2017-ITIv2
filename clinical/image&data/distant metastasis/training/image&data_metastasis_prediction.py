import shelve  # datos persistentes
import pandas as pd
import numpy as np
import seaborn as sns  # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2  # OpenCV
import glob
import staintools
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import *
from tensorflow.keras.layers import *
from functools import reduce  # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.model_selection import train_test_split  # Se importa la librería para dividir los datos en entreno y test.
from sklearn.preprocessing import MinMaxScaler  # Para escalar valores
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix  # Para realizar la matriz de confusión

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN DATOS TABULARES ------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
list_to_read = ['CNV_oncomine', 'age', 'all_oncomine', 'mutations_oncomine', 'cancer_type', 'cancer_type_detailed',
                'dfs_months', 'dfs_status', 'dict_genes', 'dss_months', 'dss_status', 'ethnicity',
                'full_length_oncomine', 'fusions_oncomine', 'muted_genes', 'CNA_genes', 'hotspot_oncomine', 'mutations',
                'CNAs', 'neoadjuvant', 'os_months', 'os_status', 'path_m_stage', 'path_n_stage', 'path_t_stage', 'sex',
                'stage', 'subtype', 'tumor_type', 'new_tumor', 'person_neoplasm_status', 'prior_diagnosis',
                'pfs_months', 'pfs_status', 'radiation_therapy']

filename = '/home/avalderas/img_slides/data/brca_tcga_pan_can_atlas_2018.out'

""" Almacenamos en una variable el diccionario de las mutaciones CNV (CNAs), en otra el diccionario de genes, que
hará falta para identificar los ID de los distintos genes a predecir y por último el diccionario de la categoría N del 
sistema de estadificación TNM: """
with shelve.open(filename) as data:
    dict_genes = data.get('dict_genes')
    age = data.get('age')
    neoadjuvant = data.get('neoadjuvant') # 1 valor nulo
    prior_diagnosis = data.get('prior_diagnosis') # 1 valor nulo
    # Datos de metástasis a distancia
    os_status = data.get('os_status')
    dfs_status = data.get('dfs_status')  # 142 valores nulos
    tumor_type = data.get('tumor_type')
    stage = data.get('stage')
    path_t_stage = data.get('path_t_stage')
    path_n_stage = data.get('path_n_stage')
    path_m_stage = data.get('path_m_stage')
    subtype = data.get('subtype')
    snv = data.get('mutations')
    cnv = data.get('CNAs')

""" Se crean dataframes individuales para cada uno de los diccionarios almacenados en cada variable de entrada y se
renombran las columnas para que todo quede más claro. Además, se crea una lista con todos los dataframes para
posteriormente unirlos todos juntos. """
df_age = pd.DataFrame.from_dict(age.items()); df_age.rename(columns = {0 : 'ID', 1 : 'Age'}, inplace = True)
df_neoadjuvant = pd.DataFrame.from_dict(neoadjuvant.items()); df_neoadjuvant.rename(columns = {0 : 'ID', 1 : 'neoadjuvant'}, inplace = True)
df_prior_diagnosis = pd.DataFrame.from_dict(prior_diagnosis.items()); df_prior_diagnosis.rename(columns = {0 : 'ID', 1 : 'prior_diagnosis'}, inplace = True)
# Columna de metástasis a distancia
df_os_status = pd.DataFrame.from_dict(os_status.items()); df_os_status.rename(columns = {0 : 'ID', 1 : 'os_status'}, inplace = True)
df_dfs_status = pd.DataFrame.from_dict(dfs_status.items()); df_dfs_status.rename(columns = {0 : 'ID', 1 : 'dfs_status'}, inplace = True)
df_tumor_type = pd.DataFrame.from_dict(tumor_type.items()); df_tumor_type.rename(columns = {0 : 'ID', 1 : 'tumor_type'}, inplace = True)
df_stage = pd.DataFrame.from_dict(stage.items()); df_stage.rename(columns = {0 : 'ID', 1 : 'stage'}, inplace = True)
df_path_t_stage = pd.DataFrame.from_dict(path_t_stage.items()); df_path_t_stage.rename(columns = {0 : 'ID', 1 : 'path_t_stage'}, inplace = True)
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_path_m_stage = pd.DataFrame.from_dict(path_m_stage.items()); df_path_m_stage.rename(columns = {0 : 'ID', 1 : 'path_m_stage'}, inplace = True)
df_subtype = pd.DataFrame.from_dict(subtype.items()); df_subtype.rename(columns = {0 : 'ID', 1 : 'subtype'}, inplace = True)
df_snv = pd.DataFrame.from_dict(snv.items()); df_snv.rename(columns = {0 : 'ID', 1 : 'SNV'}, inplace = True)
df_cnv = pd.DataFrame.from_dict(cnv.items()); df_cnv.rename(columns = {0 : 'ID', 1 : 'CNV'}, inplace = True)

""" No se incluye la columna de recidivas, puesto que reduce enormemente las muestras de pacientes con metástasis """
df_list = [df_age, df_neoadjuvant, df_prior_diagnosis, df_os_status, df_tumor_type, df_stage,
           df_path_t_stage, df_path_n_stage, df_path_m_stage, df_subtype]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

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

""" Se recoloca la columna de metástasis a distancia al lado de la de recaídas para dejar las mutaciones como las 
últimas columnas. """
cols = df_all_merge.columns.tolist()
cols = cols[:5] + cols[-1:] + cols[5:-1]
df_all_merge = df_all_merge[cols]

""" En este caso, el número de muestras de imágenes y de datos deben ser iguales. Las imágenes de las que se disponen se 
enmarcan según el sistema de estadificación TNM como N1A, N1, N2A, N2, N3A, N1MI, N1B, N3, NX, N3B, N1C o N3C según la
categoría N (extensión de cáncer que se ha diseminado a los ganglios linfáticos) de dicho sistema de estadificación.
Por tanto, en los datos tabulares tendremos que quedarnos solo con los casos donde los pacientes tengan esos valores
de la categoría 'N' y habrá que usar, por tanto, una imagen para cada paciente, para que no haya errores al repartir
los subconjuntos de datos. """
# 552 filas resultantes, como en cBioPortal:
df_all_merge = df_all_merge[(df_all_merge["path_n_stage"]!='N0') & (df_all_merge["path_n_stage"]!='NX') &
                            (df_all_merge["path_n_stage"]!='N0 (I-)') & (df_all_merge["path_n_stage"]!='N0 (I+)') &
                            (df_all_merge["path_n_stage"]!='N0 (MOL+)')]

""" Al realizar un análisis de los datos de entrada se ha visto un único valor incorrecto en la columna
'cancer_type_detailed'. Por ello se sustituye dicho valor por 'Breast Invasive Carcinoma (NOS)'. También se ha apreciado
un único valor en 'tumor_type', por lo que también se realiza un cambio de valor en dicho valor atípico. Además, se 
convierten las columnas categóricas binarias a valores de '0' y '1', para no aumentar el número de columnas: """
df_all_merge.loc[df_all_merge.tumor_type == "Infiltrating Carcinoma (NOS)", "tumor_type"] = "Mixed Histology (NOS)"
df_all_merge.loc[df_all_merge.tumor_type == "Breast Invasive Carcinoma", "tumor_type"] = "Infiltrating Ductal Carcinoma"
df_all_merge.loc[df_all_merge.neoadjuvant == "No", "neoadjuvant"] = 0; df_all_merge.loc[df_all_merge.neoadjuvant == "Yes", "neoadjuvant"] = 1
df_all_merge.loc[df_all_merge.prior_diagnosis == "No", "prior_diagnosis"] = 0; df_all_merge.loc[df_all_merge.prior_diagnosis == "Yes", "prior_diagnosis"] = 1
df_all_merge.loc[df_all_merge.os_status == "0:LIVING", "os_status"] = 0; df_all_merge.loc[df_all_merge.os_status == "1:DECEASED", "os_status"] = 1
df_all_merge.loc[df_all_merge.path_m_stage == "CM0 (I+)", "path_m_stage"] = 'M0'

""" Ahora se eliminan las filas donde haya datos nulos para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Una vez la tabla tiene las columnas deseadas se procede a codificar las columnas categóricas del dataframe a valores
numéricos mediante la técnica del 'One Hot Encoding'. Más adelante se escalarán las columnas numéricas continuas, pero
ahora se realiza esta técnica antes de hacer la repartición de subconjuntos para que no haya problemas con las columnas. """
#@ get_dummies: Aplica técnica de 'One Hot Encoding', creando columnas binarias para las columnas seleccionadas
df_all_merge = pd.get_dummies(df_all_merge, columns=["tumor_type", "stage", "path_t_stage", "path_n_stage",
                                                     "path_m_stage", "subtype"])

""" Se reordenan las columnas del dataframe para colocar las nuevas columnas numericas de datos anatomopatológicos antes
de las mutaciones """
cols = df_all_merge.columns.tolist()
df_all_merge = df_all_merge[cols]

""" Ahora se eliminan las filas donde haya datos nulos para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace = True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split. """
train_data, test_data = train_test_split(df_all_merge, test_size = 0.20, stratify = df_all_merge['distant_metastasis'])
train_data, valid_data = train_test_split(train_data, test_size = 0.15, stratify = train_data['distant_metastasis'])

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------------- SECCIÓN IMÁGENES -------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Directorios de teselas con cáncer normalizadas: """
image_dir = '/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer'

""" Se seleccionan todas las rutas de las teselas: """
cancer_dir = glob.glob(image_dir + "/img_lotes_tiles*/*") # 34421

""" Se crea una serie sobre el directorio de las imágenes con cáncer que crea un array de 1-D (columna) en el que en 
cada fila hay una ruta para cada una de las imágenes con cáncer. Posteriormente, se extrae el 'ID' de cada ruta de cada
imagen y se establecen como índices de la serie, por lo que para cada ruta de la serie, ésta tendrá como índice de fila
su 'ID' correspondiente ({TCGA-A7-A13E  C:\...\...\TCGA-A7-A13E-01Z-00-DX2.3.JPG}). 
Por último, el dataframe se une con la serie creada mediante la columna 'ID'. De esta forma, cada paciente replicará sus 
filas el número de veces que tenga una imagen distinta, es decir, que si un paciente tiene 3 imágenes, la fila de datos
de ese paciente se presenta 3 veces, teniendo en cada una de ellas una ruta de imagen distinta: """
series_img = pd.Series(cancer_dir)
series_img.index = series_img.str.extract(fr"({'|'.join(df_all_merge['ID'])})",expand = False)

train_data = train_data.join(series_img.rename('img_path'),on = 'ID')
valid_data = valid_data.join(series_img.rename('img_path'),on = 'ID')
test_data = test_data.join(series_img.rename('img_path'),on = 'ID')

""" Hay valores nulos, por lo que se ha optado por eliminar esas filas para que se pueda entrenar posteriormente la
red neuronal. Aparte de eso, se ordena el dataframe según los valores de la columna 'ID': """
train_data.dropna(inplace = True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.
valid_data.dropna(inplace = True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.
test_data.dropna(inplace = True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Una vez se tienen todas las imágenes y quitados los valores nulos, tambiés es necesario de aquellas imágenes que son
intraoperatorias. Para ello nos basamos en el archivo 'Pacientes_MGR' para eliminar algunas filas de aquellas imágenes
de algunos pacientes que no nos van a servir. """
# 1672 filas resultantes (al haber menos columnas hay menos valores nulos):
remove_img_list = ['TCGA-A2-A0EW', 'TCGA-E2-A153', 'TCGA-E2-A15A', 'TCGA-E2-A15E', 'TCGA-E9-A1N4', 'TCGA-E9-A1N5',
                   'TCGA-E9-A1N6', 'TCGA-E9-A1NC', 'TCGA-E9-A1ND', 'TCGA-E9-A1NE', 'TCGA-E9-A1NH', 'TCGA-PL-A8LX',
                   'TCGA-PL-A8LZ']

for id_img in remove_img_list:
    index_train = train_data.loc[df_all_merge['ID'] == id_img].index
    index_valid = valid_data.loc[df_all_merge['ID'] == id_img].index
    index_test = test_data.loc[df_all_merge['ID'] == id_img].index
    train_data.drop(index_train, inplace = True)
    valid_data.drop(index_valid, inplace = True)
    test_data.drop(index_test, inplace = True)

""" Se iguala el número de teselas con y sin metástasis a distancia """
# Entrenamiento
train_metastasis_tiles = train_data['distant_metastasis'].value_counts()[1] # Con metástasis a distancia
train_no_metastasis_tiles = train_data['distant_metastasis'].value_counts()[0] # Sin metástasis a distancia

if train_no_metastasis_tiles >= train_metastasis_tiles:
    difference_train = train_no_metastasis_tiles - train_metastasis_tiles
    train_data = train_data.sort_values(by = 'distant_metastasis', ascending = False)
else:
    difference_train = train_metastasis_tiles - train_no_metastasis_tiles
    train_data = train_data.sort_values(by = 'distant_metastasis', ascending = True)

train_data = train_data[:-difference_train]
print(train_data['distant_metastasis'].value_counts())

# Validación
valid_metastasis_tiles = valid_data['distant_metastasis'].value_counts()[1] # Con metástasis a distancia
valid_no_metastasis_tiles = valid_data['distant_metastasis'].value_counts()[0] # Sin metástasis a distancia

if valid_no_metastasis_tiles >= valid_metastasis_tiles:
    difference_valid = valid_no_metastasis_tiles - valid_metastasis_tiles
    valid_data = valid_data.sort_values(by = 'distant_metastasis', ascending = False)
else:
    difference_valid = valid_metastasis_tiles - valid_no_metastasis_tiles
    valid_data = valid_data.sort_values(by = 'distant_metastasis', ascending = True)

valid_data = valid_data[:-difference_valid]
print(valid_data['distant_metastasis'].value_counts())

# Test
test_metastasis_tiles = test_data['distant_metastasis'].value_counts()[1] # Con metástasis a distancia
test_no_metastasis_tiles = test_data['distant_metastasis'].value_counts()[0] # Sin metástasis a distancia

if test_no_metastasis_tiles >= test_metastasis_tiles:
    difference_test = test_no_metastasis_tiles - test_metastasis_tiles
    test_data = test_data.sort_values(by = 'distant_metastasis', ascending = False)
else:
    difference_test = test_metastasis_tiles - test_no_metastasis_tiles
    test_data = test_data.sort_values(by = 'distant_metastasis', ascending = True)

test_data = test_data[:-difference_test]
print(test_data['distant_metastasis'].value_counts())

""" Una vez ya se tienen todas las imágenes valiosas y todo perfectamente enlazado entre datos e imágenes, se definen 
las dimensiones que tendrán cada una de ellas. """
alto = int(210)  # Eje Y: 630. Nº de filas. 210 x 3 = 630
ancho = int(210)  # Eje X: 1480. Nº de columnas. 210 x 7 = 1470
canales = 3  # Imágenes a color (RGB) = 3

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN PROCESAMIENTO DE DATOS -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Se lee cada tesela de 210x210 y se añade al array correspondiente de entrenamiento, test o validación. """
train_image_data = []
valid_image_data = []
test_image_data = []

for index_normal_train, image_train in enumerate(train_data['img_path']):
    train_image_data.append(cv2.imread(image_train, cv2.IMREAD_COLOR))

for index_normal_valid, image_valid in enumerate(valid_data['img_path']):
    valid_image_data.append(cv2.imread(image_valid, cv2.IMREAD_COLOR))

for index_normal_test, image_test in enumerate(test_data['img_path']):
    test_image_data.append(cv2.imread(image_test, cv2.IMREAD_COLOR))

""" Se convierten las imágenes a un array de numpy para manipularlas con más comodidad y NO se divide el array entre 255
para escalar los píxeles entre el intervalo (0-1), ya que la red convolucional elegida para entrenar las imágenes no lo
hizo originalmente, y por tanto, se debe seguir sus mismos pasos de pre-procesamiento. Como resultado, habrá un array 
con forma (N, alto, ancho, canales). """
train_image_data = np.array(train_image_data)
valid_image_data = np.array(valid_image_data)
test_image_data = np.array(test_image_data)

""" Ya se puede eliminar de los subconjuntos la columna de imágenes, que no es útil puesto que ya han sido almacenadas 
en un array de numpy; y la columna de IDs: """
train_data = train_data.drop(['img_path', 'ID'], axis = 1)
valid_data = valid_data.drop(['img_path', 'ID'], axis = 1)
test_data = test_data.drop(['img_path', 'ID'], axis = 1)

""" Se extraen los datos de salida de la red neuronal y se guardan en otra variable. """
train_labels_metastasis = train_data.pop('distant_metastasis')
valid_labels_metastasis = valid_data.pop('distant_metastasis')
test_labels_metastasis = test_data.pop('distant_metastasis')

""" Para poder entrenar la red hace falta transformar los dataframes en arrays de numpy. """
train_data = np.asarray(train_data).astype('float32')
valid_data = np.asarray(valid_data).astype('float32')
test_data = np.asarray(test_data).astype('float32')

train_labels_metastasis = np.asarray(train_labels_metastasis).astype('float32')
valid_labels_metastasis = np.asarray(valid_labels_metastasis).astype('float32')
test_labels_metastasis = np.asarray(test_labels_metastasis).astype('float32')

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada, y se recopila la longitud de las imágenes de
entrenamiento y validacion para utilizarlas posteriormente en el entrenamiento: """
del df_all_merge, df_path_n_stage, df_list

train_image_data_len = len(train_image_data)
valid_image_data_len = len(valid_image_data)
batch_dimension = 32

""" Se pueden guardar en formato de 'numpy' las imágenes, los datos y las etiquetas de test para usarlas después de 
entrenar el modelo. """
#np.save('test_image_normalized_50', test_image_data)
#np.save('test_data_normalized_50', test_data)
#np.save('test_labels_metastasis_normalized_50', test_labels_metastasis)

""" --------------------------------------------------------------------------------------------------------------------
------------------------------------------- SECCIÓN MODELO DE RED ------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------- """
""" Se definen dos entradas para dos ramas distintas: una para la red perceptrón multicapa; y otra para la red neuronal 
convolucional """
input_data = Input(shape = (train_data.shape[1]),)
input_image = Input(shape = (alto, ancho, canales))

""" La primera rama del modelo (Perceptrón multicapa) opera con la entrada de datos: """
mlp = layers.Dense(train_data.shape[1], activation = "relu")(input_data)
mlp = layers.Dropout(0.5)(mlp)
mlp = layers.Dense(32, activation = "relu")(mlp)
#mlp = layers.Dropout(0.5)(mlp)
#mlp = layers.Dense(8, activation = "relu")(mlp)

final_mlp = keras.models.Model(inputs = input_data, outputs = mlp)

""" La segunda rama del modelo será la encargada de procesar las imágenes: """
cnn_model = keras.applications.EfficientNetB7(weights = 'imagenet', input_tensor = input_image,
                                              include_top = False, pooling = 'max')

""" Se añaden capas de clasificación después de las capas congeladas de convolución. """
all_cnn_model = cnn_model.output
all_cnn_model = layers.Flatten()(all_cnn_model)
#all_cnn_model = layers.Dense(128, activation = "relu")(all_cnn_model)
#all_cnn_model = layers.Dropout(0.5)(all_cnn_model)
#all_cnn_model = layers.Dense(32, activation = "relu")(all_cnn_model)
#all_cnn_model = layers.Dropout(0.5)(all_cnn_model)
#all_cnn_model = layers.Dense(8, activation = "relu")(all_cnn_model)

final_cnn_model = Model(inputs = cnn_model.input, outputs = all_cnn_model)

""" Se congelan todas las capas convolucionales del modelo base de la red convolucional. """
# A partir de TF 2.0 @trainable = False hace tambien ejecutar las capas BN en modo inferencia (@training = False)
for layer in cnn_model.layers:
    layer.trainable = False

""" Se combina la salida de ambas ramas. """
combined = keras.layers.concatenate([final_mlp.output, final_cnn_model.output])

""" Una vez se ha concatenado la salida de ambas ramas, se aplica dos capas densamente conectadas, la última de ellas
siendo la de la predicción final con activación 'sigmoid', puesto que la salida será binaria. """
multi_input_model = layers.Dense(16, activation="relu")(combined)
multi_input_model = layers.Dropout(0.5)(multi_input_model)
multi_input_model = layers.Dense(1, activation="sigmoid")(multi_input_model)

""" El modelo final aceptará datos numéricos/categóricos en la entrada de la red perceptrón multicapa e imágenes en la
red neuronal convolucional, de forma que a la salida solo se obtenga la predicción de la metástasis a distancia. """
model = keras.models.Model(inputs = [final_mlp.input, final_cnn_model.input], outputs = multi_input_model)

""" Hay que definir las métricas de la red y configurar los distintos hiperparámetros para entrenar la red. El modelo ya
ha sido definido anteriormente, así que ahora hay que compilarlo. Para ello se define una función de loss y un 
optimizador. Con la función de loss se estimará la 'loss' del modelo. Por su parte, el optimizador actualizará los
parámetros de la red neuronal con el objetivo de minimizar la función de 'loss'. """
# @lr: tamaño de pasos para alcanzar el mínimo global de la función de loss.
metrics = [keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'),
           keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn'),
           keras.metrics.Recall(name='recall'),  # TP / (TP + FN)
           keras.metrics.Precision(name='precision'),  # TP / (TP + FP)
           keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC-ROC'),
           keras.metrics.AUC(curve='PR', name='AUC-PR')]

model.compile(loss = 'binary_crossentropy',
              optimizer = keras.optimizers.Adam(learning_rate = 0.00001),
              metrics = metrics)
model.summary()

""" Se implementa un callback: para guardar el mejor modelo que tenga la menor 'loss' en la validación. """
checkpoint_path = '/home/avalderas/img_slides/clinical/image&data/distant metastasis/inference/models/model_image&data_metastasis_{epoch:02d}_{val_loss:.2f}.h5'
mcp_save = ModelCheckpoint(filepath = checkpoint_path, monitor = 'val_loss', mode = 'min', save_best_only = True)

""" Una vez definido el modelo, se entrena: """
model.fit(x = [train_data, train_image_data], y = train_labels_metastasis, epochs = 2, verbose = 1,
          validation_data = ([valid_data, valid_image_data], valid_labels_metastasis),
          steps_per_epoch = (train_image_data_len / batch_dimension),
          validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez el modelo ya ha sido entrenado, se resetean los generadores de data augmentation de los conjuntos de 
entrenamiento y validacion y se descongelan algunas capas convolucionales del modelo base de la red para reeentrenar
todo el modelo de principio a fin ('fine tuning'). Este es un último paso opcional que puede dar grandes mejoras o un 
rápido sobreentrenamiento y que solo debe ser realizado después de entrenar el modelo con las capas congeladas. 
Para ello, primero se descongela el modelo base."""
set_trainable = 0

for layer in cnn_model.layers:
    if layer.name == 'block2a_expand_conv':
        set_trainable = True
    if set_trainable:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

""" Es importante recompilar el modelo después de hacer cualquier cambio al atributo 'trainable', para que los cambios
se tomen en cuenta. """
model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.00001),
              loss = 'binary_crossentropy',
              metrics = metrics)
model.summary()

""" Una vez descongelado las capas convolucionales seleccionadas y compilado de nuevo el modelo, se entrena otra vez. """
neural_network = model.fit(x = [train_data, train_image_data], y = train_labels_metastasis, epochs = 20, verbose = 1,
                           validation_data = ([valid_data, valid_image_data], valid_labels_metastasis),
                           #callbacks = mcp_save,
                           steps_per_epoch = (train_image_data_len / batch_dimension),
                           validation_steps = (valid_image_data_len / batch_dimension))

"""Las métricas del entreno se guardan dentro del método 'history'. Primero, se definen las variables para usarlas 
posteriormentes para dibujar la gráfica de la 'loss'."""
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