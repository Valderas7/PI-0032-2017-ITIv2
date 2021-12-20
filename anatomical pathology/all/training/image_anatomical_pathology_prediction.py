import shelve # datos persistentes
import pandas as pd
import numpy as np
import itertools
import staintools
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
import skimage.filters as sk_filters
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
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
""" - Datos de entrada: Age, cancer_type, cancer_type_detailed, dfs_months, dfs_status, dss_months, dss_status,
ethnicity, neoadjuvant, os_months, os_status, path_m_stage. path_n_stage, path_t_stage, sex, stage, subtype.
    - Salida binaria: Presenta mutación o no en el gen 'X' (BRCA1 [ID: 672] en este caso). CNAs (CNV) y mutations (SNV). """
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
    tumor_type = data.get('tumor_type')
    stage = data.get('stage')
    path_t_stage = data.get('path_t_stage')
    path_n_stage = data.get('path_n_stage')
    path_m_stage = data.get('path_m_stage')
    subtype = data.get('subtype')

""" Se crean dataframes individuales para cada uno de los diccionarios almacenados en cada variable de entrada y se
renombran las columnas para que todo quede más claro. Además, se crea una lista con todos los dataframes para
posteriormente unirlos todos juntos. """
df_tumor_type = pd.DataFrame.from_dict(tumor_type.items()); df_tumor_type.rename(columns = {0 : 'ID', 1 : 'tumor_type'}, inplace = True)
df_stage = pd.DataFrame.from_dict(stage.items()); df_stage.rename(columns = {0 : 'ID', 1 : 'stage'}, inplace = True)
df_path_t_stage = pd.DataFrame.from_dict(path_t_stage.items()); df_path_t_stage.rename(columns = {0 : 'ID', 1 : 'path_t_stage'}, inplace = True)
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_path_m_stage = pd.DataFrame.from_dict(path_m_stage.items()); df_path_m_stage.rename(columns = {0 : 'ID', 1 : 'path_m_stage'}, inplace = True)
df_subtype = pd.DataFrame.from_dict(subtype.items()); df_subtype.rename(columns = {0 : 'ID', 1 : 'subtype'}, inplace = True)

df_list = [df_tumor_type, df_stage, df_path_t_stage, df_path_n_stage, df_path_m_stage, df_subtype]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

""" En este caso, el número de muestras de imágenes y de datos deben ser iguales. Las imágenes de las que se disponen se 
enmarcan según el sistema de estadificación TNM como N1A, N1, N2A, N2, N3A, N1MI, N1B, N3, NX, N3B, N1C o N3C según la
categoría N (extensión de cáncer que se ha diseminado a los ganglios linfáticos) de dicho sistema de estadificación.
Por tanto, en los datos tabulares tendremos que quedarnos solo con los casos donde los pacientes tengan esos valores
de la categoría 'N' y habrá que usar, por tanto, una imagen para cada paciente, para que no haya errores al repartir
los subconjuntos de datos. """
# 460 filas resultantes, como en cBioPortal:
df_all_merge = df_all_merge[(df_all_merge["path_n_stage"]!='N0') & (df_all_merge["path_n_stage"]!='NX') &
                            (df_all_merge["path_n_stage"]!='N0 (I-)') & (df_all_merge["path_n_stage"]!='N0 (I+)') &
                            (df_all_merge["path_n_stage"]!='N0 (MOL+)')]

""" Se convierten datos: """
df_all_merge.loc[df_all_merge.tumor_type == "Infiltrating Carcinoma (NOS)", "tumor_type"] = "Mixed Histology (NOS)"
df_all_merge.loc[df_all_merge.tumor_type == "Breast Invasive Carcinoma", "tumor_type"] = "Infiltrating Ductal Carcinoma"
df_all_merge.loc[df_all_merge.path_m_stage == "CM0 (I+)", "path_m_stage"] = "M0"

""" Ahora, antes de transformar las variables categóricas en numéricas, se eliminan las filas donde haya datos nulos
para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Una vez la tabla tiene las columnas deseadas se procede a codificar las columnas categóricas del dataframe a valores
numéricos mediante la técnica del 'One Hot Encoding' antes de hacer la repartición de subconjuntos para que no haya 
problemas con las columnas. """
#@ get_dummies: Aplica técnica de 'One Hot Encoding', creando columnas binarias para las columnas seleccionadas
df_all_merge = pd.get_dummies(df_all_merge, columns=["tumor_type", "stage", "path_t_stage", "path_n_stage",
                                                     "path_m_stage", "subtype"])

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento, validación y test. """
# @train_test_split: Divide en subconjuntos de datos los 'arrays' o matrices especificadas.
# @random_state: Consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
train_data, test_data = train_test_split(df_all_merge, test_size = 0.20)
train_data, valid_data = train_test_split(train_data, test_size = 0.15)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------------- SECCIÓN IMÁGENES -------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Directorios de imágenes con cáncer y sin cáncer: """
image_dir = '/tiles/TCGA_normalizadas_cáncer'
#image_dir = 'C:\\Users\\valde\Desktop\Datos_repositorio\img_slides\img_lotes'

""" Se seleccionan todas las rutas de las imágenes que tienen cáncer: """
cancer_dir = glob.glob(image_dir + "/img_lotes_tiles*/*")

""" Se crea una serie sobre el directorio de las imágenes con cáncer que crea un array de 1-D (columna) en el que en 
cada fila hay una ruta para cada una de las imágenes con cáncer. Posteriormente, se extrae el 'ID' de cada ruta de cada
imagen y se establecen como índices de la serie, por lo que para cada ruta de la serie, ésta tendrá como índice de fila
su 'ID' correspondiente ({TCGA-A7-A13E  C:\...\...\TCGA-A7-A13E-01Z-00-DX2.3.JPG}). 
Por último, el dataframe se une con la serie creada mediante la columna 'ID'. De esta forma, cada paciente replicará sus 
filas el número de veces que tenga una imagen distinta, es decir, que si un paciente tiene 3 imágenes, la fila de datos
de ese paciente se presenta 3 veces, teniendo en cada una de ellas una ruta de imagen distinta: """
series_img = pd.Series(cancer_dir)
series_img.index = series_img.str.extract(fr"({'|'.join(df_all_merge['ID'])})", expand=False)

train_data = train_data.join(series_img.rename('img_path'), on='ID')
valid_data = valid_data.join(series_img.rename('img_path'), on='ID')
test_data = test_data.join(series_img.rename('img_path'), on='ID')

""" Hay valores nulos, por lo que se ha optado por eliminar esas filas para que se pueda entrenar posteriormente la
red neuronal. Aparte de eso, se ordena el dataframe según los valores de la columna 'ID': """
train_data.dropna(inplace = True) # Mantiene el DataFrame con las entradas válidas en la misma variable.
valid_data.dropna(inplace = True) # Mantiene el DataFrame con las entradas válidas en la misma variable.
test_data.dropna(inplace = True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

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

""" Una vez ya se tienen todas las imágenes valiosas y todo perfectamente enlazado entre datos e imágenes, se definen 
las dimensiones que tendrán cada una de ellas. """
alto = int(210) # Eje Y: 630. Nº de filas. 210 x 3 = 630
ancho = int(210) # Eje X: 1480. Nº de columnas. 210 x 7 = 1470
canales = 3 # Imágenes a color (RGB) = 3

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN PROCESAMIENTO DE DATOS -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# target = staintools.read_image('/home/avalderas/img_slides/img_lotes/img_lote1_cancer/TCGA-A2-A25D-01Z-00-DX1.2.JPG')

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
en un array de numpy: """
train_data = train_data.drop(['img_path'], axis = 1)
valid_data = valid_data.drop(['img_path'], axis = 1)
test_data = test_data.drop(['img_path'], axis = 1)

""" Se extraen los datos de salida para cada dato anatomopatológico """
train_labels_tumor_type = train_data.iloc[:, 1:8]
valid_labels_tumor_type = valid_data.iloc[:, 1:8]
test_labels_tumor_type = test_data.iloc[:, 1:8]

train_labels_STAGE = train_data.iloc[:, 8:18]
valid_labels_STAGE = valid_data.iloc[:, 8:18]
test_labels_STAGE = test_data.iloc[:, 8:18]

train_labels_pT = train_data.iloc[:, 18:28]
valid_labels_pT = valid_data.iloc[:, 18:28]
test_labels_pT = test_data.iloc[:, 18:28]

train_labels_pN = train_data.iloc[:, 28:39]
valid_labels_pN = valid_data.iloc[:, 28:39]
test_labels_pN = test_data.iloc[:, 28:39]

train_labels_pM = train_data.iloc[:, 39:42]
valid_labels_pM = valid_data.iloc[:, 39:42]
test_labels_pM = test_data.iloc[:, 39:42]

train_labels_IHQ = train_data.iloc[:, 42:]
valid_labels_IHQ = valid_data.iloc[:, 42:]
test_labels_IHQ = test_data.iloc[:, 42:]

""" Para poder entrenar la red hace falta transformar los dataframes en arrays de numpy. """
train_labels_tumor_type = np.asarray(train_labels_tumor_type).astype('float32')
train_labels_STAGE = np.asarray(train_labels_STAGE).astype('float32')
train_labels_pT = np.asarray(train_labels_pT).astype('float32')
train_labels_pN = np.asarray(train_labels_pN).astype('float32')
train_labels_pM = np.asarray(train_labels_pM).astype('float32')
train_labels_IHQ = np.asarray(train_labels_IHQ).astype('float32')

valid_labels_tumor_type = np.asarray(valid_labels_tumor_type).astype('float32')
valid_labels_STAGE = np.asarray(valid_labels_STAGE).astype('float32')
valid_labels_pT = np.asarray(valid_labels_pT).astype('float32')
valid_labels_pN = np.asarray(valid_labels_pN).astype('float32')
valid_labels_pM = np.asarray(valid_labels_pM).astype('float32')
valid_labels_IHQ = np.asarray(valid_labels_IHQ).astype('float32')

test_labels_tumor_type = np.asarray(test_labels_tumor_type).astype('float32')
test_labels_STAGE = np.asarray(test_labels_STAGE).astype('float32')
test_labels_pT = np.asarray(test_labels_pT).astype('float32')
test_labels_pN = np.asarray(test_labels_pN).astype('float32')
test_labels_pM = np.asarray(test_labels_pM).astype('float32')
test_labels_IHQ = np.asarray(test_labels_IHQ).astype('float32')

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada, y se recopila la longitud de las imagenes de
entrenamiento y validacion para utilizarlas posteriormente en el entrenamiento: """
del df_all_merge, df_path_n_stage, df_list, train_data

train_image_data_len = len(train_image_data) #print(train_image_data_len)
valid_image_data_len = len(valid_image_data)
batch_dimension = 32

""" Se pueden guardar en formato de 'numpy' las imágenes y las etiquetas de test para usarlas después de entrenar la red
neuronal convolucional. """
# np.save('test_image', test_image_data)
# np.save('test_labels_tumor_type', test_labels_tumor_type)
# np.save('test_labels_STAGE', test_labels_STAGE)
# np.save('test_labels_pT', test_labels_pT)
# np.save('test_labels_pN', test_labels_pN)
# np.save('test_labels_pM', test_labels_pM)
# np.save('test_labels_IHQ', test_labels_IHQ)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (CNN) -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" En esta ocasión, se crea un modelo secuencial para la red neuronal convolucional que será la encargada de procesar
todas las imágenes: """
base_model = keras.applications.EfficientNetB7(weights = 'imagenet', input_tensor = Input(shape=(alto, ancho, canales)),
                                               include_top = False, pooling = 'max')
all_model = base_model.output
all_model = layers.Flatten()(all_model)
all_model = layers.Dense(256)(all_model)
all_model = layers.Dropout(0.5)(all_model)
all_model = layers.Dense(32)(all_model)
all_model = layers.Dropout(0.5)(all_model)

tumor_type = layers.Dense(train_labels_tumor_type.shape[1], activation = "softmax", name= 'tumor_type')(all_model)
STAGE = layers.Dense(train_labels_STAGE.shape[1], activation = "softmax", name = 'STAGE')(all_model)
pT = layers.Dense(train_labels_pT.shape[1], activation = "softmax", name= 'pT')(all_model)
pN = layers.Dense(train_labels_pN.shape[1], activation = "softmax", name= 'pN')(all_model)
pM = layers.Dense(train_labels_pM.shape[1], activation = "softmax", name= 'pM')(all_model)
IHQ = layers.Dense(train_labels_IHQ.shape[1], activation = "softmax", name= 'IHQ')(all_model)

model = keras.models.Model(inputs = base_model.input, outputs = [tumor_type, STAGE, pT, pN, pM, IHQ])

""" Se congelan todas las capas convolucionales del modelo base """
# A partir de TF 2.0 @trainable = False hace tambien ejecutar las capas BN en modo inferencia (@training = False)
for layer in base_model.layers:
    layer.trainable = False

""" Hay que definir las métricas de la red y configurar los distintos hiperparámetros para entrenar la red. El modelo ya
ha sido definido anteriormente, así que ahora hay que compilarlo. Para ello se define una función de loss y un 
optimizador. Con la función de loss se estimará la 'loss' del modelo. Por su parte, el optimizador actualizará los
parámetros de la red neuronal con el objetivo de minimizar la función de 'loss'. """
# @lr: tamaño de pasos para alcanzar el mínimo global de la función de loss.
metrics = [keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'),
           keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn'),
           keras.metrics.Recall(name='recall'), # TP / (TP + FN)
           keras.metrics.Precision(name='precision'), # TP / (TP + FP)
           keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC-ROC'),
           keras.metrics.AUC(curve = 'PR', name = 'AUC-PR')]

model.compile(loss = {'tumor_type': 'categorical_crossentropy', 'STAGE': 'categorical_crossentropy',
                      'pT': 'categorical_crossentropy', 'pN': 'categorical_crossentropy',
                      'pM': 'categorical_crossentropy', 'IHQ': 'categorical_crossentropy'},
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)
model.summary()

""" Se implementan varios callbacks para guardar el mejor modelo. """
checkpoint_path = '/anatomical_pathology_data/image/all_data/inference/models/model_image_anatomopathologic_{epoch:02d}_{val_loss:.2f}.h5'
mcp_save = ModelCheckpoint(filepath = checkpoint_path, monitor = 'val_loss', mode = 'min')

""" Una vez definido el modelo, se entrena: """
model.fit(x = train_image_data, y = {'tumor_type': train_labels_tumor_type, 'STAGE': train_labels_STAGE,
                                'pT': train_labels_pT, 'pN': train_labels_pN, 'pM': train_labels_pM,
                                'IHQ': train_labels_IHQ},
          epochs = 3, verbose = 1, validation_data = (valid_image_data, {'tumor_type': valid_labels_tumor_type,
                                                                         'STAGE': valid_labels_STAGE, 
                                                                         'pT': valid_labels_pT, 'pN': valid_labels_pN, 
                                                                         'pM': valid_labels_pM, 'IHQ': valid_labels_IHQ}), 
          steps_per_epoch = (train_image_data_len / batch_dimension),
          validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez el modelo ya ha sido entrenado, se descongelan algunas capas convolucionales del modelo base de la red para 
reeentrenar el modelo ('fine tuning'). Este es un último paso opcional que puede dar grandes mejoras o un rápido 
sobreentrenamiento y que solo debe ser realizado después de entrenar el modelo con las capas congeladas """
set_trainable = 0

for layer in base_model.layers:
    if layer.name == 'block3a_expand_conv':
        set_trainable = True
    if set_trainable:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

""" Es importante recompilar el modelo después de hacer cualquier cambio al atributo 'trainable', para que los cambios
se tomen en cuenta """
model.compile(loss = {'tumor_type': 'categorical_crossentropy', 'STAGE': 'categorical_crossentropy',
                      'pT': 'categorical_crossentropy', 'pN': 'categorical_crossentropy',
                      'pM': 'categorical_crossentropy', 'IHQ': 'categorical_crossentropy'},
              optimizer = keras.optimizers.Adam(learning_rate = 0.00001),
              metrics = metrics)
model.summary()

""" Una vez descongeladas las capas convolucionales seleccionadas y compilado de nuevo el modelo, se entrena otra vez. """
neural_network = model.fit(x = train_image_data, y = {'tumor_type': train_labels_tumor_type, 'STAGE': train_labels_STAGE, 
                                                      'pT': train_labels_pT, 'pN': train_labels_pN, 
                                                      'pM': train_labels_pM, 'IHQ': train_labels_IHQ}, 
                           epochs = 100, verbose = 1, validation_data = (valid_image_data,
                                                                         {'tumor_type': valid_labels_tumor_type,
                                                                          'STAGE': valid_labels_STAGE, 
                                                                          'pT': valid_labels_pT, 'pN': valid_labels_pN, 
                                                                          'pM': valid_labels_pM, 
                                                                          'IHQ': valid_labels_IHQ}),
                           #callbacks = mcp_save,
                           steps_per_epoch = (train_image_data_len / batch_dimension), 
                           validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data, [test_labels_tumor_type, test_labels_STAGE, test_labels_pT, test_labels_pN,
                                             test_labels_pM, test_labels_IHQ], verbose = 0)

print("\n'Loss' del tipo histológico en el conjunto de prueba: {:.2f}\n""Sensibilidad del tipo histológico en el "
      "conjunto de prueba: {:.2f}\n""Precisión del tipo histológico en el conjunto de prueba: {:.2f}\n""Especifidad del "
      "tipo histológico en el conjunto de prueba: {:.2f}\n""Exactitud del tipo histológico en el conjunto de prueba: "
      "{:.2f} %\n""AUC-ROC del tipo histológico en el conjunto de prueba: {:.2f}\n""AUC-PR del tipo histológico en el "
      "conjunto de prueba: {:.2f}".format(results[1], results[11], results[12], results[9]/(results[9]+results[8]), 
                                          results[13] * 100, results[14], results[15]))
if results[11] > 0 or results[12] > 0:
    print("Valor-F del tipo histológico en el conjunto de prueba: {:.2f}".format((2 * results[11] * results[12]) /
                                                                                    (results[11] + results[12])))

print("\n'Loss' del estadio anatomopatológico en el conjunto de prueba: {:.2f}\n""Sensibilidad del estadio "
      "anatomopatológico en el conjunto de prueba: {:.2f}\n""Precisión del estadio anatomopatológico en el conjunto de "
      "prueba: {:.2f}\n""Especifidad del estadio anatomopatológico en el conjunto de prueba: {:.2f} \n""Exactitud del "
      "estadio anatomopatológico en el conjunto de prueba: {:.2f} %\n""AUC-ROC del estadio anatomopatológico en el "
      "conjunto de prueba: {:.2f}\n""AUC-PR del estadio anatomopatológico "
      "en el conjunto de prueba: {:.2f}".format(results[2], results[20], results[21], 
                                                results[18]/(results[18]+results[17]), results[22] * 100, results[23], 
                                                results[24]))
if results[20] > 0 or results[21] > 0:
    print("Valor-F del estadio anatomopatológico en el conjunto de prueba: {:.2f}".format((2 * results[20] * results[21]) /
                                                                                            (results[20] + results[21])))

print("\n'Loss' del parámetro 'T' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'T' en el conjunto de "
      "prueba: {:.2f}\n""Precisión del parámetro 'T' en el conjunto de prueba: {:.2f}\n""Especifidad del parámetro 'T' "
      "en el conjunto de prueba: {:.2f}\n""Exactitud del parámetro 'T' en el conjunto de prueba: {:.2f} %\n""AUC-ROC "
      "del parámetro 'T' en el conjunto de prueba: {:.2f}\n""AUC-PR del parámetro 'T' en el conjunto de prueba: "
      "{:.2f}".format(results[3], results[29], results[30], results[27]/(results[27]+results[26]), results[31] * 100,
                      results[32], results[33]))

if results[29] > 0 or results[30] > 0:
    print("Valor-F del parámetro 'T' en el conjunto de prueba: {:.2f}".format((2 * results[29] * results[30]) /
                                                                                (results[29] + results[30])))

print("\n'Loss' del parámetro 'N' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'N' en el conjunto de "
      "prueba: {:.2f}\n""Precisión del parámetro 'N' en el conjunto de prueba: {:.2f}\n""Especifidad del parámetro 'N' "
      "en el conjunto de prueba: {:.2f} \n""Exactitud del parámetro 'N' en el conjunto de prueba: {:.2f} %\n""AUC-ROC "
      "del parámetro 'N' en el conjunto de prueba: {:.2f}\n""AUC-PR del parámetro 'N' en el conjunto de prueba: "
      "{:.2f}".format(results[4], results[38], results[39], results[36]/(results[36]+results[35]), results[40] * 100,
                      results[41], results[42]))

if results[38] > 0 or results[39] > 0:
    print("Valor-F del parámetro 'N' en el conjunto de prueba: {:.2f}".format((2 * results[38] * results[39]) /
                                                                                (results[38] + results[39])))

print("\n'Loss' del parámetro 'M' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'M' en el conjunto de "
      "prueba: {:.2f}\n""Precisión del parámetro 'M' en el conjunto de prueba: {:.2f}\n""Especifidad del parámetro 'M' "
      "en el conjunto de prueba: {:.2f} \n""Exactitud del parámetro 'M' en el conjunto de prueba: {:.2f} %\n""AUC-ROC "
      "del parámetro 'M' en el conjunto de prueba: {:.2f}\n""AUC-PR del parámetro 'M' en el conjunto de prueba: "
      "{:.2f}".format(results[5], results[47], results[48], results[45]/(results[45]+results[44]), results[49] * 100,
                      results[50], results[51]))

if results[47] > 0 or results[48] > 0:
    print("Valor-F del parámetro 'M' en el conjunto de prueba: {:.2f}".format((2 * results[47] * results[48]) /
                                                                                (results[47] + results[48])))

print("\n'Loss' del subtipo molecular en el conjunto de prueba: {:.2f}\n""Sensibilidad del subtipo molecular en el "
      "conjunto de prueba: {:.2f}\n""Precisión del subtipo molecular en el conjunto de prueba: {:.2f}\n""Especifidad del "
      "subtipo molecular en el conjunto de prueba: {:.2f}\n""Exactitud del subtipo molecular en el conjunto de prueba: "
      "{:.2f} %\n""AUC-ROC del subtipo molecular en el conjunto de prueba: {:.2f}\n""AUC-PR del subtipo molecular en el "
      "conjunto de prueba: {:.2f}".format(results[6], results[56], results[57], results[54]/(results[54]+results[53]),
                                          results[58] * 100, results[59], results[60]))

if results[56] > 0 or results[57] > 0:
    print("Valor-F del subtipo molecular en el conjunto de prueba: {:.2f}".format((2 * results[56] * results[57]) /
                                                                                    (results[56] + results[57])))

"""Las métricas del entreno se guardan dentro del método 'history'. Primero, se definen las variables para usarlas 
posteriormentes para dibujar la gráficas de la 'loss' del entrenamiento y validación de cada iteración."""
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
# Tipo histológico
y_true_tumor_type = []
for label_test_tumor_type in test_labels_tumor_type:
    y_true_tumor_type.append(np.argmax(label_test_tumor_type))

y_true_tumor_type = np.array(y_true_tumor_type)
y_pred_tumor_type = np.argmax(model.predict(test_image_data)[0], axis = 1)

matrix_tumor_type = confusion_matrix(y_true_tumor_type, y_pred_tumor_type, labels = [0, 1, 2, 3, 4, 5, 6])
matrix_tumor_type_classes = ['IDC', 'ILC', 'Medullary', 'Metaplastic', 'Mixed (NOS)', 'Mucinous', 'Other']

# Estadio anatomopatológico
y_true_STAGE = []
for label_test_STAGE in test_labels_STAGE:
    y_true_STAGE.append(np.argmax(label_test_STAGE))

y_true_STAGE = np.array(y_true_STAGE)
y_pred_STAGE = np.argmax(model.predict(test_image_data)[1], axis = 1)

matrix_STAGE = confusion_matrix(y_true_STAGE, y_pred_STAGE, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Calcula (pero no dibuja) la matriz de confusión
matrix_STAGE_classes = ['Stage IB', 'Stage II', 'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IIIA', 'Stage IIIB',
                        'Stage IIIC', 'Stage IV', 'STAGE X']

# pT
y_true_pT = []
for label_test_pT in test_labels_pT:
    y_true_pT.append(np.argmax(label_test_pT))

y_true_pT = np.array(y_true_pT)
y_pred_pT = np.argmax(model.predict(test_image_data)[2], axis = 1)

matrix_pT = confusion_matrix(y_true_pT, y_pred_pT, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pT_classes = ['T1', 'T1A', 'T1B', 'T1C', 'T2', 'T2B', 'T3', 'T4', 'T4B', 'T4D']

# pN
y_true_pN = []
for label_test_pN in test_labels_pN:
    y_true_pN.append(np.argmax(label_test_pN))

y_true_pN = np.array(y_true_pN)
y_pred_pN = np.argmax(model.predict(test_image_data)[3], axis = 1)

matrix_pN = confusion_matrix(y_true_pN, y_pred_pN, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pN_classes = ['N1', 'N1A', 'N1B', 'N1C', 'N1MI', 'N2', 'N2A', 'N3', 'N3A', 'N3B', 'N3C']

# pM
y_true_pM = []
for label_test_pM in test_labels_pM:
    y_true_pM.append(np.argmax(label_test_pM))

y_true_pM = np.array(y_true_pM)
y_pred_pM = np.argmax(model.predict(test_image_data)[4], axis = 1)

matrix_pM = confusion_matrix(y_true_pM, y_pred_pM, labels = [0, 1, 2])
matrix_pM_classes = ['M0', 'M1', 'MX']

# IHQ
y_true_IHQ = []
for label_test_IHQ in test_labels_IHQ:
    y_true_IHQ.append(np.argmax(label_test_IHQ))

y_true_IHQ = np.array(y_true_IHQ)
y_pred_IHQ = np.argmax(model.predict(test_image_data)[5], axis = 1)

matrix_IHQ = confusion_matrix(y_true_IHQ, y_pred_IHQ, labels = [0, 1, 2, 3, 4]) # Calcula (pero no dibuja) la matriz de confusión
matrix_IHQ_classes = ['Basal', 'Her-2', 'Luminal A', 'Luminal B', 'Normal']

""" Función para mostrar por pantalla la matriz de confusión multiclase con todas las clases de subtipos moleculares """
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión', cmap = plt.cm.Blues):
    """ Imprime y dibuja la matriz de confusión. Se puede normalizar escribiendo el parámetro `normalize=True`. """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
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

plot_confusion_matrix(matrix_tumor_type, classes = matrix_tumor_type_classes, title = 'Matriz de confusión del tipo '
                                                                                      'histológico')
plt.show()

plot_confusion_matrix(matrix_STAGE, classes = matrix_STAGE_classes, title = 'Matriz de confusión del estadio '
                                                                            'anatomopatológico')
plt.show()

plot_confusion_matrix(matrix_pT, classes = matrix_pT_classes, title = 'Matriz de confusión del parámetro "T"')
plt.show()

plot_confusion_matrix(matrix_pN, classes = matrix_pN_classes, title = 'Matriz de confusión del parámetro "N"')
plt.show()

plot_confusion_matrix(matrix_pM, classes = matrix_pM_classes, title = 'Matriz de confusión del parámetro "M"')
plt.show()

plot_confusion_matrix(matrix_IHQ, classes = matrix_IHQ_classes, title = 'Matriz de confusión del subtipo molecular')
plt.show()

""" Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases.
Para implementarlas, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva: """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
from scipy import interp

# Tipo histologico
y_pred_prob_tumor_type = model.predict(test_image_data)[0]

fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(matrix_tumor_type_classes)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_tumor_type[:, i], y_pred_prob_tumor_type[:, i])
    auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_tumor_type.ravel(), y_pred_prob_tumor_type.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for tumor type')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_tumor_type_classes)):
    precision[i], recall[i], _ = precision_recall_curve(test_labels_tumor_type[:, i], y_pred_prob_tumor_type[:, i])
    auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_tumor_type.ravel(),
                                                                y_pred_prob_tumor_type.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for tumor type')
plt.legend(loc = 'best')
plt.show()

# Estadio anatomopatológico
y_pred_prob_STAGE = model.predict(test_image_data)[1]

fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(matrix_STAGE_classes)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_STAGE[:, i], y_pred_prob_STAGE[:, i])
    auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_STAGE.ravel(), y_pred_prob_STAGE.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for STAGE')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_STAGE_classes)):
    precision[i], recall[i], _ = precision_recall_curve(test_labels_STAGE[:, i], y_pred_prob_STAGE[:, i])
    auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_STAGE.ravel(),
                                                                y_pred_prob_STAGE.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for STAGE')
plt.legend(loc = 'best')
plt.show()

# pT
y_pred_prob_pT = model.predict(test_image_data)[2]

fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(matrix_pT_classes)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_pT[:, i], y_pred_prob_pT[:, i])
    auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_pT.ravel(), y_pred_prob_pT.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("AUC-ROC curve for 'pT' parameter")
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_pT_classes)):
    precision[i], recall[i], _ = precision_recall_curve(test_labels_pT[:, i], y_pred_prob_pT[:, i])
    auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_pT.ravel(),
                                                                y_pred_prob_pT.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("AUC-PR curve for 'pT' parameter")
plt.legend(loc = 'best')
plt.show()

# pN
y_pred_prob_pN = model.predict(test_image_data)[3]

fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(matrix_pN_classes)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_pN[:, i], y_pred_prob_pN[:, i])
    auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_pN.ravel(), y_pred_prob_pN.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("AUC-ROC curve for 'pN' parameter")
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_pN_classes)):
    precision[i], recall[i], _ = precision_recall_curve(test_labels_pN[:, i], y_pred_prob_pN[:, i])
    auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_pN.ravel(),
                                                                y_pred_prob_pN.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("AUC-PR curve for 'pN' parameter")
plt.legend(loc = 'best')
plt.show()

# pM
y_pred_prob_pM = model.predict(test_image_data)[4]

fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(matrix_pM_classes)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_pM[:, i], y_pred_prob_pM[:, i])
    auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_pM.ravel(), y_pred_prob_pM.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("AUC-ROC curve for 'pM' parameter")
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_pT_classes)):
    precision[i], recall[i], _ = precision_recall_curve(test_labels_pM[:, i], y_pred_prob_pM[:, i])
    auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_pM.ravel(),
                                                                y_pred_prob_pM.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("AUC-PR curve for 'pM' parameter")
plt.legend(loc = 'best')
plt.show()

# Subtipo molecular
y_pred_prob_IHQ = model.predict(test_image_data)[5]

fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(matrix_IHQ_classes)):
    fpr[i], tpr[i], _ = roc_curve(test_labels_IHQ[:, i], y_pred_prob_IHQ[:, i])
    auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_IHQ.ravel(), y_pred_prob_IHQ.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title("AUC-ROC curve for molecular subtype")
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(matrix_pT_classes)):
    precision[i], recall[i], _ = precision_recall_curve(test_labels_IHQ[:, i], y_pred_prob_IHQ[:, i])
    auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_IHQ.ravel(),
                                                                y_pred_prob_IHQ.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("AUC-PR curve for molecular subtype")
plt.legend(loc = 'best')
plt.show()