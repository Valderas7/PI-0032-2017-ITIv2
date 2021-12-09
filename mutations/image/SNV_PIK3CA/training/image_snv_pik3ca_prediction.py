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
# 1) Imágenes
# 2) Datos de mutaciones: SNV, CNV-A, CNV-D.
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
    path_n_stage = data.get('path_n_stage')
    snv = data.get('mutations')
    cnv = data.get('CNAs')

""" Se crea un dataframe para el diccionario de mutaciones SNV y otro para el diccionario de la categoría N del sistema
de estadificación TNM. Posteriormente, se renombran las dos columnas para que todo quede más claro y se fusionan ambos
dataframes. En una columna tendremos el ID del paciente, en otra las distintas mutaciones SNV y en la otra la 
categoría N para dicho paciente. """
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns={0: 'ID', 1: 'path_n_stage'}, inplace=True)
df_snv = pd.DataFrame.from_dict(snv.items()); df_snv.rename(columns={0: 'ID', 1: 'SNV'}, inplace=True)
df_cnv = pd.DataFrame.from_dict(cnv.items()); df_cnv.rename(columns={0: 'ID', 1: 'CNV'}, inplace=True)

df_list = [df_path_n_stage, df_snv, df_cnv]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left, right: pd.merge(left, right, on=['ID'], how='left'), df_list)

""" Ahora se va a encontrar cuales son los ID de los genes que nos interesa. Para empezar se carga el archivo excel 
donde aparecen todos los genes con mutaciones que interesan estudiar usando 'openpyxl' y creamos una lista para
los genes CNV."""
mutations_target = pd.read_excel('/home/avalderas/img_slides/excel_genesOCA&inibica_patients/Panel_OCA.xlsx',
                                 usecols='B:C', engine='openpyxl')

snv = mutations_target.loc[mutations_target['Scope'] != 'CNV', 'Gen']

# SNV
snv_list = []
for gen_snv in snv:
    if gen_snv not in snv_list:
        snv_list.append(gen_snv)

""" Ahora se recopilan en una lista los distintos IDs de los genes a estudiar. """
id_snv_list = []

key_list = list(dict_genes.keys())
val_list = list(dict_genes.values())

# SNV
for gen_snv in snv_list:
    position = val_list.index(gen_snv)  # Número
    id_gen_snv = (key_list[position])  # Número
    id_snv_list.append(id_gen_snv)  # Se añaden todos los IDs en la lista vacía

""" Se recopila los índices de las distintas filas donde aparecen las mutaciones 'CNV' de los genes seleccionados (tanto 
de amplificación como deleción), y se añaden a la lista de listas correspondiente (la de amplificación o la de deleción). """
# SNV
list_gen_snv = [[] for ID in range(len(id_snv_list))]

for index, id_snv in enumerate(id_snv_list):  # Para cada ID del gen SNV de la lista...
    for index_row, row in enumerate(df_all_merge['SNV']):  # Para cada fila dentro de la columna 'SNV'...
        for mutation in row:  # Para cada mutación dentro de cada fila...
            if mutation[1] == id_snv:  # Si el ID de la mutación es el mismo que el ID de la lista de genes...
                list_gen_snv[index].append(index_row)  # Se almacena el índice de la fila en la lista de listas

""" Una vez se tienen almacenados los índices de las filas donde se producen las mutaciones CNV, hay que crear distintas 
columnas para cada uno de los genes objetivo, para asi mostrar la informacion de uno en uno. De esta forma, habra dos 
columnas distintas para cada gen CNV a estudiar (amplificacion y delecion). Ademas, se recopilan las columnas creadas en 
listas (dos para las columnas de mutaciones CNV). """
columns_list_snv = []

# SNV:
df_all_merge.drop(['SNV'], axis=1, inplace=True)
for gen_snv in snv_list:
    df_all_merge['SNV_' + gen_snv] = 0
    columns_list_snv.append('SNV_' + gen_snv)

""" Una vez han sido creadas las columnas, se añade un '1' en aquellas filas donde el paciente tiene mutación sobre el
gen seleccionado. Se utiliza para ello los índices recogidos anteriormente en las respectivas listas de listas. De esta
forma, iterando sobre la lista de columnas creadas y mediante los distintos indices de cada sublista, se consigue
colocar un '1' en aquella filas donde el paciente tiene la mutacion especificada en el gen especificado. """
# SNV
i_snv = 0
for column_snv in columns_list_snv:
    for index_snv_sublist in list_gen_snv[i_snv]:
        df_all_merge.loc[index_snv_sublist, column_snv] = 1
    i_snv += 1

""" En este caso, el número de muestras de imágenes y de datos deben ser iguales. Las imágenes de las que se disponen se 
enmarcan según el sistema de estadificación TNM como N1A, N1, N2A, N2, N3A, N1MI, N1B, N3, NX, N3B, N1C o N3C según la
categoría N (extensión de cáncer que se ha diseminado a los ganglios linfáticos) de dicho sistema de estadificación.
Por tanto, en los datos tabulares tendremos que quedarnos solo con los casos donde los pacientes tengan esos valores
de la categoría 'N' y habrá que usar, por tanto, una imagen para cada paciente, para que no haya errores al repartir
los subconjuntos de datos. """
# 552 filas resultantes, como en cBioPortal:
df_all_merge = df_all_merge[(df_all_merge["path_n_stage"] != 'N0') & (df_all_merge["path_n_stage"] != 'NX') &
                            (df_all_merge["path_n_stage"] != 'N0 (I-)') & (df_all_merge["path_n_stage"] != 'N0 (I+)') &
                            (df_all_merge["path_n_stage"] != 'N0 (MOL+)')]

""" Se eliminan todas las columnas de mutaciones excepto la de SNV_PIK3CA """
df_all_merge = df_all_merge[['ID', 'SNV_PIK3CA']]
df_all_merge = df_all_merge.sort_values(by='SNV_PIK3CA', ascending = False)
df_all_merge = df_all_merge[:-210] # Ahora hay el mismo número de pacientes con mutación y sin mutación
df_all_merge.dropna(inplace=True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
train_data, test_data = train_test_split(df_all_merge, test_size = 0.20, stratify = df_all_merge['SNV_PIK3CA'])
train_data, valid_data = train_test_split(train_data, test_size = 0.15, stratify = train_data['SNV_PIK3CA'])

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------------- SECCIÓN IMÁGENES -------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Directorios de teselas con cáncer normalizadas: """
image_dir = '/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer'

""" Se seleccionan todas las rutas de las teselas: """
cancer_dir = glob.glob(image_dir + "/img_lotes_tiles*/*")  # 34421

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
train_data.dropna(inplace=True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.
valid_data.dropna(inplace=True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.
test_data.dropna(inplace=True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.

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
    train_data.drop(index_train, inplace=True)
    valid_data.drop(index_valid, inplace=True)
    test_data.drop(index_test, inplace=True)

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
para escalar los píxeles entre el intervalo (0-1), ya que la red convolucional elegida para entrenar las imagenes no lo
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

""" Se extraen las etiquetas de salida para cada la mutación SNV de TP53 """
train_labels_pik3ca = train_data.iloc[:, -1]
valid_labels_pik3ca = valid_data.iloc[:, -1]
test_labels_pik3ca = test_data.iloc[:, -1]

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada, y se recopila la longitud de las imágenes de
entrenamiento y validacion para utilizarlas posteriormente en el entrenamiento: """
del df_all_merge, df_path_n_stage, df_list

train_image_data_len = len(train_image_data)  # print(train_image_data_len)
valid_image_data_len = len(valid_image_data)
batch_dimension = 32

""" Para poder entrenar la red hace falta transformar las tablas en arrays. Para ello se utiliza 'numpy'. Las imágenes 
YA están convertidas en 'arrays' numpy """
train_labels_pik3ca = np.asarray(train_labels_pik3ca).astype('float32')
valid_labels_pik3ca = np.asarray(valid_labels_pik3ca).astype('float32')
test_labels_pik3ca = np.asarray(test_labels_pik3ca).astype('float32')

""" Se pueden guardar en formato de 'numpy' las imágenes y las etiquetas de test para usarlas después de entrenar la red
neuronal convolucional. """
#np.save('test_image', test_image_data)
#np.save('test_labels_pik3ca', test_labels_pik3ca)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL CONVOLUCIONAL ---------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" En esta ocasión, se crea un modelo secuencial para la red neuronal convolucional que será la encargada de procesar
todas las imágenes: """
base_model = keras.applications.EfficientNetB7(weights='imagenet', input_tensor=Input(shape=(alto, ancho, canales)),
                                               include_top=False, pooling='max')

all_model = base_model.output
all_model = layers.Flatten()(all_model)
all_model = layers.Dense(256)(all_model)
all_model = layers.Dropout(0.5)(all_model)
all_model = layers.Dense(32)(all_model)
all_model = layers.Dropout(0.5)(all_model)
pik3ca = layers.Dense(1, activation="sigmoid", name='pik3ca')(all_model)

model = Model(inputs=base_model.input, outputs = pik3ca)

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
           keras.metrics.Recall(name='recall'),  # TP / (TP + FN)
           keras.metrics.Precision(name='precision'),  # TP / (TP + FP)
           keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='AUC-ROC'),
           keras.metrics.AUC(curve='PR', name='AUC-PR')]

model.compile(loss= 'binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate = 0.00001),
              metrics=metrics)
model.summary()

""" Se implementa un callbacks para guardar el modelo cada época. """
checkpoint_path = '/home/avalderas/img_slides/mutations/image/SNV_PIK3CA/inference/models/model_image_pik3ca_{epoch:02d}_{val_loss:.2f}.h5'
mcp_save = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min')

""" Una vez definido el modelo, se entrena: """
model.fit(x = train_image_data, y = train_labels_pik3ca,
          epochs = 1, verbose = 1, validation_data = (valid_image_data, valid_labels_pik3ca),
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
se tomen en cuenta. """
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.00001),
              loss='binary_crossentropy',
              metrics=metrics)
model.summary()

""" Una vez descongelado las capas convolucionales seleccionadas y compilado de nuevo el modelo, se entrena otra vez. """
neural_network = model.fit(x = train_image_data, y = train_labels_pik3ca,
                           epochs = 15, verbose = 1, validation_data = (valid_image_data, valid_labels_pik3ca),
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
plt.figure()  # Crea o activa una figura