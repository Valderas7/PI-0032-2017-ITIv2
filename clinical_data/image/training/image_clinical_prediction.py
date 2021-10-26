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
# 1) Imágenes.
# 2) Datos clínicos: Metástasis a distancia, supervivencia, recaída.
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
    os_status = data.get('os_status')
    dfs_status = data.get('dfs_status')  # 142 valores nulos
    path_m_stage = data.get('path_m_stage')

""" Se crea un dataframe para el diccionario de mutaciones SNV y otro para el diccionario de la categoría N del sistema
de estadificación TNM. Posteriormente, se renombran las dos columnas para que todo quede más claro y se fusionan ambos
dataframes. En una columna tendremos el ID del paciente, en otra las distintas mutaciones SNV y en la otra la 
categoría N para dicho paciente. """
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns={0: 'ID', 1: 'path_n_stage'}, inplace=True)
df_os_status = pd.DataFrame.from_dict(os_status.items()); df_os_status.rename(columns={0: 'ID', 1: 'os_status'}, inplace=True)
df_dfs_status = pd.DataFrame.from_dict(dfs_status.items()); df_dfs_status.rename(columns={0: 'ID', 1: 'dfs_status'}, inplace=True)
df_path_m_stage = pd.DataFrame.from_dict(path_m_stage.items()); df_path_m_stage.rename(columns={0: 'ID', 1: 'path_m_stage'}, inplace=True)

df_list = [df_path_n_stage, df_os_status, df_dfs_status, df_path_m_stage]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left, right: pd.merge(left, right, on=['ID'], how='left'), df_list)

""" Las imágenes de las que se disponen se enmarcan según el sistema de estadificación TNM como N1A, N1, N2A, N2, N3A, 
N1MI, N1B, N3, NX, N3B, N1C o N3C según la categoría N (extensión de cáncer que se ha diseminado a los ganglios 
linfáticos) de dicho sistema de estadificación.
Por tanto, en los datos tabulares tendremos que quedarnos solo con los casos donde los pacientes tengan esos valores
de la categoría 'N'. """
# 552 filas resultantes, como en cBioPortal:
df_all_merge = df_all_merge[(df_all_merge["path_n_stage"] != 'N0') & (df_all_merge["path_n_stage"] != 'NX') &
                            (df_all_merge["path_n_stage"] != 'N0 (I-)') & (df_all_merge["path_n_stage"] != 'N0 (I+)') &
                            (df_all_merge["path_n_stage"] != 'N0 (MOL+)')]

""" Se convierten las columnas de pocos valores en columnas binarias: """
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

""" Ahora se eliminan las filas donde haya datos nulos para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split. """
train_data, test_data = train_test_split(df_all_merge, test_size = 0.20)
train_data, valid_data = train_test_split(train_data, test_size = 0.20)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------------- SECCIÓN IMÁGENES -------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Directorios de teselas con cáncer normalizadas: """
image_dir = 'home/avalderas/img_slides/split_images_into_tiles/TCGA_normalizadas_cáncer/'

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

""" Se extraen los datos de salida para cada dato clínico """
train_labels_survival = train_data.iloc[:, 2]
valid_labels_survival = valid_data.iloc[:, 2]
test_labels_survival = test_data.iloc[:, 2]

train_labels_relapse = train_data.iloc[:, 3]
valid_labels_relapse = valid_data.iloc[:, 3]
test_labels_relapse = test_data.iloc[:, 3]

train_labels_metastasis = train_data.iloc[:, -1]
valid_labels_metastasis = valid_data.iloc[:, -1]
test_labels_metastasis = test_data.iloc[:, -1]

""" Para poder entrenar la red hace falta transformar los dataframes en arrays de numpy. """
train_labels_survival = np.asarray(train_labels_survival).astype('float32')
train_labels_relapse = np.asarray(train_labels_relapse).astype('float32')
train_labels_metastasis = np.asarray(train_labels_metastasis).astype('float32')

valid_labels_survival = np.asarray(valid_labels_survival).astype('float32')
valid_labels_relapse = np.asarray(valid_labels_relapse).astype('float32')
valid_labels_metastasis = np.asarray(valid_labels_metastasis).astype('float32')

test_labels_survival = np.asarray(test_labels_survival).astype('float32')
test_labels_relapse = np.asarray(test_labels_relapse).astype('float32')
test_labels_metastasis = np.asarray(test_labels_metastasis).astype('float32')

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada, y se recopila la longitud de las imágenes de
entrenamiento y validacion para utilizarlas posteriormente en el entrenamiento: """
del df_all_merge, df_path_n_stage, df_list

train_image_data_len = len(train_image_data)
valid_image_data_len = len(valid_image_data)
batch_dimension = 32

""" Se pueden guardar en formato de 'numpy' las imágenes y las etiquetas de test para usarlas después de entrenar la red
neuronal convolucional. """
# np.save('test_image', test_image_data)
# np.save('test_labels_survival', test_labels_survival)
# np.save('test_labels_relapse', test_labels_relapse)
# np.save('test_labels_metastasis', test_labels_metastasis)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL CONVOLUCIONAL ---------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" En esta ocasión, se crea un modelo secuencial para la red neuronal convolucional que será la encargada de procesar
todas las imágenes: """
base_model = keras.applications.EfficientNetB7(weights='imagenet', input_tensor=Input(shape=(alto, ancho, canales)),
                                               include_top=False, pooling='max')

all_model = base_model.output
all_model = layers.Flatten()(all_model)
all_model = layers.Dense(512)(all_model)
all_model = layers.Dropout(0.5)(all_model)
all_model = layers.Dense(128)(all_model)
all_model = layers.Dropout(0.5)(all_model)

survival = layers.Dense(1, activation = "sigmoid", name = 'survival')(all_model)
relapse = layers.Dense(1, activation = "sigmoid", name = 'relapse')(all_model)
metastasis = layers.Dense(1, activation = "sigmoid", name = 'metastasis')(all_model)

model = Model(inputs = base_model.input, outputs = [survival, relapse, metastasis])

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

model.compile(loss = {'survival': 'binary_crossentropy', 'relapse': 'binary_crossentropy',
                      'metastasis': 'binary_crossentropy'},
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)

model.summary()

""" Se implementa un callback: para guardar el mejor modelo que tenga la mayor F1-Score en la validación. """
checkpoint_path = '/home/avalderas/img_slides/clinical_data/image/inference/test_data&models/model_clinical_image.h5'
mcp_save = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

""" Una vez definido el modelo, se entrena: """
model.fit(x = train_image_data, y = {'survival': train_labels_survival, 'relapse': train_labels_relapse,
                                     'metastasis': train_labels_metastasis},
          epochs = 1, verbose = 1, validation_data = (valid_image_data, {'survival': valid_labels_survival,
                                                                         'relapse': valid_labels_relapse,
                                                                         'metastasis': valid_labels_metastasis}),
          steps_per_epoch = (train_image_data_len / batch_dimension),
          validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez el modelo ya ha sido entrenado, se resetean los generadores de data augmentation de los conjuntos de 
entrenamiento y validacion y se descongelan algunas capas convolucionales del modelo base de la red para reeentrenar
todo el modelo de principio a fin ('fine tuning'). Este es un último paso opcional que puede dar grandes mejoras o un 
rápido sobreentrenamiento y que solo debe ser realizado después de entrenar el modelo con las capas congeladas. 
Para ello, primero se descongela el modelo base."""
set_trainable = 0

for layer in base_model.layers:
    if layer.name == 'block5a_expand_conv':
        set_trainable = True
    if set_trainable:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

""" Es importante recompilar el modelo después de hacer cualquier cambio al atributo 'trainable', para que los cambios
se tomen en cuenta. """
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.0001),
              loss={'snv': 'binary_crossentropy', 'cnv_a': 'binary_crossentropy', 'cnv_normal': 'binary_crossentropy',
                    'cnv_d': 'binary_crossentropy'},
              metrics=metrics)
model.summary()

""" Una vez descongelado las capas convolucionales seleccionadas y compilado de nuevo el modelo, se entrena otra vez. """
neural_network = model.fit(x = train_image_data, y = {'survival': train_labels_survival,
                                                      'relapse': train_labels_relapse,
                                                      'metastasis': train_labels_metastasis},
                           epochs = 1, verbose = 1, validation_data = (valid_image_data,
                                                                       {'survival': valid_labels_survival,
                                                                        'relapse': valid_labels_relapse,
                                                                        'metastasis': valid_labels_metastasis}),
                           #callbacks = mcp_save,
                           steps_per_epoch = (train_image_data_len / batch_dimension),
                           validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data, [test_labels_survival, test_labels_relapse, test_labels_metastasis],
                         verbose = 0)

print("\n'Loss' de la supervivencia en el conjunto de prueba: {:.2f}\n""Sensibilidad de la supervivencia en el conjunto "
      "de prueba: {:.2f}\n""Precisión de la supervivencia en el conjunto de prueba: {:.2f}\n""Especifidad de la "
      "supervivencia en el conjunto de prueba: {:.2f} \n""Exactitud de la supervivencia en el conjunto de prueba: {:.2f} "
      "%\n""AUC-ROC de la supervivencia en el conjunto de prueba: {:.2f}\n""AUC-PR de la supervivencia en el conjunto de "
      "prueba: {:.2f}".format(results[1], results[8], results[9], results[6]/(results[6]+results[5]),
                              results[10] * 100, results[11], results[12]))

if results[8] > 0 or results[9] > 0:
    print("Valor-F de la supervivencia en el conjunto de prueba: {:.2f}".format((2 * results[8] * results[9]) /
                                                                                (results[8] + results[9])))

print("\n'Loss' de recidivas en el conjunto de prueba: {:.2f}\n""Sensibilidad de recidivas en el conjunto de prueba: "
      "{:.2f}\n""Precisión de recidivas en el conjunto de prueba: {:.2f}\n""Especifidad de recidivas en el conjunto de "
      "prueba: {:.2f} \n""Exactitud de recidivas en el conjunto de prueba: {:.2f} %\n""AUC-ROC de recidivas en el "
      "conjunto de prueba: {:.2f}\n""AUC-PR de recidivas en el conjunto de prueba: {:.2f}".format(results[2],
                                                                                                  results[17],
                                                                                                  results[18],
                                                                                                  results[15]/(results[15]+results[14]),
                                                                                                  results[19] * 100,
                                                                                                  results[20], results[21]))
if results[17] > 0 or results[18] > 0:
    print("Valor-F de recidivas en el conjunto de prueba: {:.2f}".format((2 * results[17] * results[18]) /
                                                                         (results[17] + results[18])))

print("\n'Loss' de la metástasis a distancia en el conjunto de prueba: {:.2f}\n""Sensibilidad de la metástasis a "
      "distancia en el conjunto de prueba: {:.2f}\n""Precisión de la metástasis a distancia en el conjunto de prueba: "
      "{:.2f}\n""Especifidad de la metástasis a distancia en el conjunto de prueba: {:.2f} \n""Exactitud de la "
      "metástasis a distancia en el conjunto de prueba: {:.2f} %\n""AUC-ROC de la metástasis a distancia en el conjunto "
      "de prueba: {:.2f}\n""AUC-PR de la metástasis a distancia en el conjunto de prueba: {:.2f}".format(results[3],
                                                                                                         results[26],
                                                                                                         results[27],
                                                                                                         results[22]/(results[22]+results[21]),
                                                                                                         results[28] * 100,
                                                                                                         results[29],
                                                                                                         results[30]))

if results[26] > 0 or results[27] > 0:
    print("Valor-F de la metástasis a distancia en el conjunto de prueba: {:.2f}".format((2 * results[26] * results[27]) /
                                                                                         (results[26] + results[27])))

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
plt.show() # Se muestran todas las gráficas

""" -------------------------------------------------------------------------------------------------------------------
------------------------------------------- SECCIÓN DE EVALUACIÓN  ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos.
Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
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

# Supervivencia
y_true_survival = test_labels_survival
y_pred_survival = np.round(model.predict(test_tabular_data)[0])

matrix_survival = confusion_matrix(y_true_survival, y_pred_survival, labels = [0, 1])
matrix_survival_classes = ['Viviendo', 'Fallecida']

plot_confusion_matrix(matrix_survival, classes = matrix_survival_classes, title = 'Matriz de confusión de supervivencia '
                                                                                  'del paciente')
plt.show()

# Recidivas
y_true_relapse = test_labels_relapse
y_pred_relapse = np.round(model.predict(test_tabular_data)[1])

matrix_relapse = confusion_matrix(y_true_relapse, y_pred_relapse, labels = [0, 1])
matrix_relapse_classes = ['Sin Recidiva', 'Con Recidiva']

plot_confusion_matrix(matrix_relapse, classes = matrix_relapse_classes, title = 'Matriz de confusión de recidivas')
plt.show()

# Metástasis a distancia
y_true_metastasis = test_labels_metastasis
y_pred_metastasis = np.round(model.predict(test_tabular_data)[2])

matrix_metastasis = confusion_matrix(y_true_metastasis, y_pred_metastasis, labels = [0, 1])
matrix_metastasis_classes = ['Sin Metástasis', 'Con Metástasis']

plot_confusion_matrix(matrix_metastasis, classes = matrix_metastasis_classes, title = 'Matriz de confusión de metástasis '
                                                                                      'a distancia')
plt.show()

""" Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases.
Para implementarlas, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva: """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve

y_pred_prob_survival = model.predict(test_image_data)[0].ravel()
y_pred_prob_relapse = model.predict(test_image_data)[1].ravel()
y_pred_prob_metastasis = model.predict(test_image_data)[2].ravel()

# Supervivencia
fpr, tpr, thresholds = roc_curve(test_labels_survival, y_pred_prob_survival)
auc_roc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for survival prediction')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision, recall, threshold_survival = precision_recall_curve(test_labels_survival, y_pred_prob_survival)
auc_pr = auc(recall, precision)

plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for survival prediction')
plt.legend(loc = 'best')
plt.show()

# Recidivas
fpr, tpr, thresholds_relapse = roc_curve(test_labels_relapse, y_pred_prob_relapse)
auc_roc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for relapse prediction')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision, recall, threshold = precision_recall_curve(test_labels_relapse, y_pred_prob_relapse)
auc_pr = auc(recall, precision)

plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for relapse prediction')
plt.legend(loc = 'best')
plt.show()

# Metastasis
fpr, tpr, thresholds_metastasis = roc_curve(test_labels_metastasis, y_pred_prob_metastasis)
auc_roc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for distant metastasis prediction')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision, recall, threshold_metastasis = precision_recall_curve(test_labels_metastasis, y_pred_prob_metastasis)
auc_pr = auc(recall, precision)

plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for distant metastasis prediction')
plt.legend(loc = 'best')
plt.show()