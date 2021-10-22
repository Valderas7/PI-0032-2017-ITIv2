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
    path_n_stage = data.get('path_n_stage')
    subtype = data.get('subtype')

""" Se crean dataframes individuales para cada uno de los diccionarios almacenados en cada variable de entrada y se
renombran las columnas para que todo quede más claro. Además, se crea una lista con todos los dataframes para
posteriormente unirlos todos juntos. """
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_subtype = pd.DataFrame.from_dict(subtype.items()); df_subtype.rename(columns = {0 : 'ID', 1 : 'subtype'}, inplace = True)

df_list = [df_path_n_stage, df_subtype]

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

""" Ahora, antes de transformar las variables categóricas en numéricas, se eliminan las filas donde haya datos nulos
para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento, validación y test. """
# @train_test_split: Divide en subconjuntos de datos los 'arrays' o matrices especificadas.
# @random_state: Consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
train_data, test_data = train_test_split(df_all_merge, test_size = 0.20, stratify = df_all_merge['subtype'])
train_data, valid_data = train_test_split(train_data, test_size = 0.20, stratify = train_data['subtype'])

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------------- SECCIÓN IMÁGENES -------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Directorios de imágenes con cáncer y sin cáncer: """
image_dir = '/home/avalderas/img_slides/img_lotes'
#image_dir = 'C:\\Users\\valde\Desktop\Datos_repositorio\img_slides\img_lotes'

""" Se seleccionan todas las rutas de las imágenes que tienen cáncer: """
cancer_dir = glob.glob(image_dir + "/img_lote*_cancer/*")

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
train_data.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.
valid_data.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.
test_data.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

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
alto = int(630) # Eje Y: 630. Nº de filas. 210 x 3 = 630
ancho = int(1470) # Eje X: 1480. Nº de columnas. 210 x 7 = 1470
canales = 3 # Imágenes a color (RGB) = 3

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN PROCESAMIENTO DE DATOS -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Se establece una imagen como la imagen objetivo respecto a la que normalizar el color, estandarizando también el
brillo para mejorar el cálculo """
# @StainNormalizer: Instancia para normalizar el color de la imagen mediante el metodo de normalizacion especificado
normalizer = staintools.StainNormalizer(method = 'vahadane')
target = staintools.read_image('/home/avalderas/img_slides/img_lotes/img_lote1_cancer/TCGA-A2-A25D-01Z-00-DX1.2.JPG')
target = staintools.LuminosityStandardizer.standardize(target)
target = normalizer.fit(target)

""" Se extrae la columna 'subtype' del dataframe de todos los subconjuntos, ya que ésta es la salida del modelo que se 
va a entrenar. """
train_labels = train_data['subtype']
valid_labels = valid_data['subtype']
test_labels = test_data['subtype']

""" Se binarizan los cinco subtipos de la columna 'subtype' para que de esta forma los datos sean válidos para la red. """
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
valid_labels = lb.transform(valid_labels)
test_labels = lb.transform(test_labels)

""" Se dividen las imágenes en teselas del mismo tamaño (en este caso ancho = 7 x 120 y alto = 3 x 210), por lo que al 
final se tienen un total de 21 teselas por imagen. Por otra parte, también se multiplican las etiquetas de salida 
dependiendo del número de teselas en el que se ha dividido la imagen. De esta forma, cada imagen tendrá N teselas, y 
también N filas en las etiquetas de salida, para que así cada tesela esté etiquetada correctamente dependiendo de la 
imagen de la que provenía. """
train_image_tile = []
train_image_data = []
train_labels_tile = []

valid_image_tile = []
valid_image_data = []
valid_labels_tile = []

test_image_tile = []
test_image_data = []
test_labels_tile = []

for index_normal_train, image_train in enumerate(train_data['img_path']):
    train_image_resize = staintools.read_image(image_train)
    train_image_resize = staintools.LuminosityStandardizer.standardize(train_image_resize)
    train_image_resize = normalizer.transform(train_image_resize)
    train_image_resize = cv2.resize(train_image_resize, (ancho, alto), interpolation = cv2.INTER_CUBIC)
    train_tiles = [train_image_resize[x:x + 210, y:y + 210] for x in range(0, train_image_resize.shape[0], 210) for y in
                   range(0, train_image_resize.shape[1], 210)]

    for train_tile in train_tiles:
        if not np.all(train_tile == 255):
            train_image_tile.append(train_tile)
            train_image_data.append(train_tile)

    train_tile_df_labels = pd.DataFrame(train_labels[index_normal_train, :]).transpose()
    train_tile_df_labels = pd.DataFrame(np.repeat(train_tile_df_labels.values, len(train_image_tile), axis = 0),
                                     columns = lb.classes_)
    train_image_tile.clear()
    train_labels_tile.append(train_tile_df_labels)

train_labels = pd.concat(train_labels_tile, ignore_index=True)

for index_normal_valid, image_valid in enumerate(valid_data['img_path']):
    valid_image_resize = staintools.read_image(image_valid)
    valid_image_resize = staintools.LuminosityStandardizer.standardize(valid_image_resize)
    valid_image_resize = normalizer.transform(valid_image_resize)
    valid_image_resize = cv2.resize(valid_image_resize, (ancho, alto), interpolation = cv2.INTER_CUBIC)
    valid_tiles = [valid_image_resize[x:x + 210, y:y + 210] for x in range(0, valid_image_resize.shape[0], 210) for y in
                   range(0, valid_image_resize.shape[1], 210)]

    for valid_tile in valid_tiles:
        if not np.all(valid_tile == 255):
            valid_image_tile.append(valid_tile)
            valid_image_data.append(valid_tile)

    valid_tile_df_labels = pd.DataFrame(valid_labels[index_normal_valid, :]).transpose()
    valid_tile_df_labels = pd.DataFrame(np.repeat(valid_tile_df_labels.values, len(valid_image_tile), axis = 0),
                                        columns = lb.classes_)
    valid_image_tile.clear()
    valid_labels_tile.append(valid_tile_df_labels)

valid_labels = pd.concat(valid_labels_tile, ignore_index=True)

for index_normal_test, image_test in enumerate(test_data['img_path']):
    test_image_resize = staintools.read_image(image_test)
    test_image_resize = staintools.LuminosityStandardizer.standardize(test_image_resize)
    test_image_resize = normalizer.transform(test_image_resize)
    test_image_resize = cv2.resize(test_image_resize, (ancho, alto), interpolation = cv2.INTER_CUBIC)
    test_tiles = [test_image_resize[x:x + 210, y:y + 210] for x in range(0, test_image_resize.shape[0], 210) for y in
                  range(0, test_image_resize.shape[1], 210)]

    for test_tile in test_tiles:
        test_image_tile.append(test_tile)
        test_image_data.append(test_tile)

    test_tile_df_labels = pd.DataFrame(test_labels[index_normal_test, :]).transpose()
    test_tile_df_labels = pd.DataFrame(np.repeat(test_tile_df_labels.values, len(test_image_tile), axis = 0),
                                        columns = lb.classes_)
    test_image_tile.clear()
    test_labels_tile.append(test_tile_df_labels)

test_labels = pd.concat(test_labels_tile, ignore_index=True)

""" Se convierten las imágenes a un array de numpy para poderlas introducir posteriormente en el modelo de red. Además,
se divide todo el array de imágenes entre 255 para escalar los píxeles en el intervalo (0-1). Como resultado, habrá un 
array con forma (X, alto, ancho, canales). """
train_image_data = np.array(train_image_data)
valid_image_data = np.array(valid_image_data)
test_image_data = np.array(test_image_data) # / 255.0)

""" Para poder entrenar la red hace falta transformar las tablas en arrays. Para ello se utiliza 'numpy'. Las imágenes 
YA están convertidas en 'arrays' numpy """
train_labels = np.asarray(train_labels).astype('float32')
valid_labels = np.asarray(valid_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada, y se recopila la longitud de las imagenes de
entrenamiento y validacion para utilizarlas posteriormente en el entrenamiento: """
del df_all_merge, df_path_n_stage, df_list, train_data

train_image_data_len = len(train_image_data)
valid_image_data_len = len(valid_image_data)
batch_dimension = 32

""" Se pueden guardar en formato de 'numpy' las imágenes y las etiquetas de test para usarlas después de entrenar la red
neuronal convolucional. """
#np.save('test_image', test_image_data)
#np.save('test_labels', test_labels)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (CNN) -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" En esta ocasión, se crea un modelo secuencial para la red neuronal convolucional que será la encargada de procesar
todas las imágenes: """
base_model = keras.applications.EfficientNetB7(weights = 'imagenet', input_tensor = Input(shape=(210, 210, canales)),
                                              include_top = False, pooling = 'max')
all_model = base_model.output
all_model = layers.Flatten()(all_model)
all_model = layers.Dense(256)(all_model)
all_model = layers.Dropout(0.5)(all_model)
all_model = layers.Dense(16)(all_model)
all_model = layers.Dropout(0.5)(all_model)
all_model = layers.Dense(len(lb.classes_), activation = 'softmax')(all_model)
model = keras.models.Model(inputs = base_model.input, outputs = all_model)

""" Se congelan todas las capas convolucionales del modelo base """
# A partir de TF 2.0 @trainable = False hace tambien ejecutar las capas BN en modo inferencia (@training = False)
for layer in base_model.layers:
    layer.trainable = False

""" Se instancian generadores de técnicas de 'data augmentation'. Como se puede comprobar, se van a instanciar dos 
generadores: uno para el conjunto de entrenamiento donde se realizan técnicas de rotación, volteo de imágenes, etc; y 
otro para el conjunto de validación, que se mantendrá igual que estaba anteriormente, sin modificaciones. """
trainAug = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, zoom_range= 0.2,
                              shear_range= 0.2, width_shift_range= 0.2, height_shift_range= 0.2, rotation_range= 20)
valAug = ImageDataGenerator()

""" Se utilizan las imágenes de cada uno de los distintos subconjuntos de datos para los distintos generadores creados 
arriba. Como ya se ha descrito anteriormente, las imágenes de entrenamiento serán usadas en el generador de técnicas de 
'data augmentation', mientras que las imágenes de validación y test no sufrirán cambio alguno. """
trainGen = trainAug.flow(x = train_image_data, y = train_labels, batch_size = 32)
valGen = valAug.flow(x = valid_image_data, y = valid_labels, batch_size = 32, shuffle = False)
#testGen = valAug.flow(x = test_image_data, y = test_labels, batch_size = 32, shuffle = False)

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
           keras.metrics.AUC(curve = 'PR', name='AUC-PR')]

model.compile(loss = 'categorical_crossentropy', # Esta función de loss suele usarse para clasificación binaria.
              optimizer = keras.optimizers.Adam(learning_rate = 0.001),
              metrics = metrics)
model.summary()

""" Se implementa un callback: para guardar el mejor modelo que tenga la mayor F1-Score en la validación. """
checkpoint_path = '/home/avalderas/img_slides/subtype_classification/inference/image/test_data&models/model_image_subtype_epoch{epoch:02d}.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = True,
                           monitor= '(2 * val_recall * val_precision) / (val_recall + val_precision)', mode = 'max')

""" Esto se hace para que al hacer el entrenamiento, los pesos de las distintas salidas se balanceen, ya que el conjunto
de datos que se tratan en este problema es muy imbalanceado. """
from sklearn.utils.class_weight import compute_class_weight

y_integers = np.argmax(train_labels, axis = 1)
class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_integers), y = y_integers)
d_class_weights = dict(enumerate(class_weights)) # {0: 1.4780, 1: 2.055238, 2: 0.40186, 3: 0.85... etc}

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
model.fit(trainGen, epochs = 3, verbose = 1, steps_per_epoch = (train_image_data_len / batch_dimension),
          class_weight = d_class_weights, validation_data = valGen,
          validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez el modelo ya ha sido entrenado, se resetean los generadores de data augmentation de los conjuntos de 
entrenamiento y validacion y se descongelan algunas capas convolucionales del modelo base de la red para reeentrenar
todo el modelo de principio a fin ('fine tuning'). Este es un último paso opcional que puede dar grandes mejoras o un 
rápido sobreentrenamiento y que solo debe ser realizado después de entrenar el modelo con las capas congeladas. 
Para ello, primero se descongela el modelo base."""
trainGen.reset()
valGen.reset()

for layer in base_model.layers[-150:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

""" Es importante recompilar el modelo después de hacer cualquier cambio al atributo 'trainable', para que los cambios
se tomen en cuenta """
model.compile(loss = 'categorical_crossentropy', # Esta función de loss suele usarse para clasificación binaria.
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)
model.summary()

""" Una vez descongeladas las capas convolucionales seleccionadas y compilado de nuevo el modelo, se entrena otra vez. """
neural_network = model.fit(trainGen, epochs = 100, verbose = 1, validation_data = valGen,
                           steps_per_epoch = (train_image_data_len / batch_dimension), class_weight = d_class_weights,
                           #callbacks = mcp_save,
                           validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data, test_labels, verbose = 0)
print("\n'Loss' del subtipo molecular en el conjunto de prueba: {:.2f}\n""Sensibilidad del subtipo molecular en el "
      "conjunto de prueba: {:.2f}\nPrecisión del subtipo molecular en el conjunto de prueba: {:.2f}\n""Especifidad del "
      "subtipo molecular en el conjunto de prueba: {:.2f} \nExactitud del subtipo molecular en el conjunto de prueba: "
      "{:.2f} %\nEl AUC-ROC del subtipo molecular en el conjunto de prueba es de: {:.2f}\nEl AUC-PR del subtipo "
      "molecular en el conjunto de prueba es de: {:.2f}".format(results[0], results[5], results[6],
                                                                results[3]/(results[3]+results[2]), results[7] * 100,
                                                                results[8], results[9]))

if results[5] > 0 or results[6] > 0:
    print("Valor-F del subtipo molecular en el en el conjunto de prueba: {:.2f}".format((2 * results[5] * results[6]) /
                                                                                (results[5] + results[6])))


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
conjunto de datos de test que se definió anteriormente al repartir los datos. """
# @suppress=True: Muestra los números con representación de coma fija
# @predict: Genera predicciones para nuevas entradas
print("\nGenera predicciones para 10 muestras")
print("Clase de las salidas:\n\r", test_labels[:10])
print("\n")
np.set_printoptions(precision=3, suppress=True)
print("Predicciones:\n", np.round(model.predict(test_image_data[:10])))
print("\n")

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
y_true = []
for label_test in test_labels:
    y_true.append(np.argmax(label_test))

y_true = np.array(y_true)
y_pred = np.argmax(model.predict(test_image_data), axis = 1)

matrix = confusion_matrix(y_true, y_pred, labels = [0, 1, 2, 3, 4]) # Calcula (pero no dibuja) la matriz de confusión
matrix_classes = ['Basal', 'Her-2', 'Luminal A', 'Luminal B', 'Normal']

""" Función para mostrar por pantalla la matriz de confusión multiclase con todas las clases de subtipos moleculares """
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión', cmap = plt.cm.Reds):
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

plot_confusion_matrix(matrix, classes = matrix_classes, title = 'Matriz de confusión del subtipo molecular')
plt.show()

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
y_pred_prob = model.predict(test_image_data)

""" Una vez calculadas las dos puntuaciones, se dibuja la curva micro-promedio. Esto es mejor que dibujar una curva para 
cada una de las clases que hay en el problema. """
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(lb.classes_)):
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
for i in range(len(lb.classes_)):
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