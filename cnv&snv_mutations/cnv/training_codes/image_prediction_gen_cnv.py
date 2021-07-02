import shelve # datos persistentes
import pandas as pd
import numpy as np
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

""" Almacenamos en una variable el diccionario de las mutaciones SNV (mutations), en otra el diccionario de genes, que
hará falta para identificar los ID de los distintos genes a predecir y por último el diccionario de la categoría N del 
sistema de estadificación TNM: """
with shelve.open(filename) as data:
    dict_genes = data.get('dict_genes')
    path_n_stage = data.get('path_n_stage')
    cnv = data.get('CNAs')

""" Se crea un dataframe para el diccionario de mutaciones SNV y otro para el diccionario de la categoría N del sistema
de estadificación TNM. Posteriormente, se renombran las dos columnas para que todo quede más claro y se fusionan ambos
dataframes. En una columna tendremos el ID del paciente, en otra las distintas mutaciones SNV y en la otra la 
categoría N para dicho paciente. """
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_cnv = pd.DataFrame.from_dict(cnv.items()); df_cnv.rename(columns = {0 : 'ID', 1 : 'CNV'}, inplace = True)

df_list = [df_path_n_stage, df_cnv]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

""" Ahora se va a encontrar cual es el ID del gen que se quiere predecir. Para ello se crean dos variables para
crear una lista de claves y otra de los valores del diccionario de genes. Se extrae el índice del gen en la lista de
valores y posteriormente se usa ese índice para buscar con qué clave (ID) se corresponde en la lista de claves. """
key_list = list(dict_genes.keys())
val_list = list(dict_genes.values())

position = val_list.index('ERBB2') # Número AQUÍ ESPECIFICAMOS EL GEN CUYA MUTACIÓN SNV SE QUIERE PREDECIR
id_gen = (key_list[position]) # Número

""" Se hace un bucle sobre la columna de mutaciones del dataframe. Así, se busca en cada mutación de cada fila para ver
en que filas se puede encontrar el ID del gen que se quiere predecir. Se almacenan en una lista los índices de las filas
donde se encuentra ese ID. """
list_gen = []

for index, row in enumerate (df_all_merge['CNV']): # Para cada fila...
    for mutation in row: # Para cada mutación de cada fila...
        if mutation[1] == id_gen:
            list_gen.append(index)

""" Una vez se tienen almacenados los índices de las filas donde se produce esa mutación, como la salida de la red será
binaria, se transforman todos los valores de la columna 'mutations' a '0' (no hay mutación del gen específico). Y una 
vez hecho esto, ya se añaden los '1' (sí hay mutación del gen específico) en las filas cuyos índices estén almacenados 
en la lista 'list_gen'. """
df_all_merge['CNV'] = 0

for index in list_gen:
    df_all_merge.loc[index, 'CNV'] = 1

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

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
train_data, test_data = train_test_split(df_all_merge, test_size = 0.20, stratify = df_all_merge['CNV'],
                                         random_state = 42)
""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------------- SECCIÓN IMÁGENES -------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Directorios de imágenes con cáncer y sin cáncer: """
image_dir = '/home/avalderas/img_slides/img_lotes'

""" Se seleccionan todas las rutas de las imágenes que tienen cáncer: """
cancer_dir = glob.glob(image_dir + "/img_lote*_cancer/*") # 1702 imágenes con cáncer en total

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
test_data = test_data.join(series_img.rename('img_path'), on='ID')

""" Hay valores nulos, por lo que se ha optado por eliminar esas filas para que se pueda entrenar posteriormente la
red neuronal. Aparte de eso, se ordena el dataframe según los valores de la columna 'ID': """
train_data.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.
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
    index_test = test_data.loc[df_all_merge['ID'] == id_img].index
    train_data.drop(index_train, inplace=True)
    test_data.drop(index_test, inplace=True)

""" Una vez ya se tienen todas las imágenes valiosas y todo perfectamente enlazado entre datos e imágenes, se definen 
las dimensiones que tendrán cada una de ellas. """
# IMPORTANTE: La anchura no puede ser más alta que la altura.
alto = 100 # 630
ancho = 100 # 1480
canales = 3 # Imágenes a color (RGB) = 3

""" Se leen y se redimensionan posteriormente las imágenes a las dimensiones especificadas arriba: """
pre_train_image_data = [] # Lista con las imágenes redimensionadas
test_image_data = [] # Lista con las imágenes redimensionadas del subconjunto de test

for imagen_train in train_data['img_path']:
    pre_train_image_data.append(cv2.resize(cv2.imread(imagen_train, cv2.IMREAD_COLOR), (alto, ancho),
                                           interpolation=cv2.INTER_CUBIC))

for imagen_test in test_data['img_path']:
    test_image_data.append(cv2.resize(cv2.imread(imagen_test, cv2.IMREAD_COLOR), (alto, ancho),
                                           interpolation=cv2.INTER_CUBIC))

train_image_data = []

for image in pre_train_image_data:
    train_image_data.append(image)
    rotate = iaa.Affine(rotate=(-20, 20), mode= 'edge')
    train_image_data.append(rotate.augment_image(image))
    gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
    train_image_data.append(gaussian_noise.augment_image(image))
    crop = iaa.Crop(percent=(0, 0.3))
    train_image_data.append(crop.augment_image(image))
    shear = iaa.Affine(shear=(0, 40), mode= 'edge')
    train_image_data.append(shear.augment_image(image))
    flip_hr = iaa.Fliplr(p=1.0)
    train_image_data.append(flip_hr.augment_image(image))
    flip_vr = iaa.Flipud(p=1.0)
    train_image_data.append(flip_vr.augment_image(image))
    contrast = iaa.GammaContrast(gamma=2.0)
    train_image_data.append(contrast.augment_image(image))
    scale_im = iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
    train_image_data.append(scale_im.augment_image(image))

""" Se convierten las imágenes a un array de numpy para manipularlas con más comodidad y se divide el array entre 255
para escalar los píxeles entre el intervalo (0-1). Como resultado, habrá un array con forma (471, alto, ancho, canales). """
train_image_data = (np.array(train_image_data) / 255.0)
test_image_data = (np.array(test_image_data) / 255.0)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN PROCESAMIENTO DE DATOS -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Una vez se tienen hechos los recortes de imágenes, se procede a replicar las filas de ambos subconjuntos de datos
para que el número de imágenes utilizadas y el número de filas del marco de datos sea el mismo: """
train_data = pd.DataFrame(np.repeat(train_data.values, 9, axis=0), columns=train_data.columns)

""" Una vez ya se tienen las imágenes convertidas en arrays y en el orden establecido por cada paciente, se puede
extraer del dataframe la columna 'SNV', que será la salida de la red:"""
train_labels = train_data.pop('CNV')
test_labels = test_data.pop('CNV')

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada: """
del df_all_merge, df_path_n_stage, df_list

""" Para poder entrenar la red hace falta transformar las tablas en arrays. Para ello se utiliza 'numpy'. Las imágenes 
YA están convertidas en 'arrays' numpy """
train_labels = np.asarray(train_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL CONVOLUCIONAL ---------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" En esta ocasión, se crea un modelo secuencial para la red neuronal convolucional que será la encargada de procesar
todas las imágenes: """
model = keras.models.Sequential()
model.add(layers.SeparableConv2D(32, (3, 3), input_shape = (alto, ancho, canales),padding="same", activation='relu'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.SeparableConv2D(64, (3, 3), padding="same", activation='relu'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.SeparableConv2D(64, (3, 3), padding="same", activation='relu'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation= 'sigmoid'))
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
           keras.metrics.BinaryAccuracy(name='accuracy')]

""" Se implementa un callback: para guardar el mejor modelo que tenga la mayor sensibilidad en la validación. """
checkpoint_path = 'model_cnv_image_myc_epoch{epoch:02d}.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = False)

model.compile(loss = 'binary_crossentropy', # Esta función de loss suele usarse para clasificación binaria.
              optimizer = keras.optimizers.Adam(learning_rate = 0.001),
              metrics = metrics)

""" Se calculan los pesos de las dos clases del problema: """
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
class_weight_dict = dict(enumerate(class_weights))

""" Una vez definido el modelo, se entrena: """
neural_network = model.fit(x = train_image_data,  # Datos de entrada.
                           y = train_labels,  # Datos de salida.
                           class_weight=class_weight_dict,
                           epochs = 7,
                           callbacks= mcp_save,
                           verbose = 2,
                           batch_size= 32,
                           validation_split = 0.2) # Datos de validación.

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data,test_labels, verbose = 0)

print("\n'Loss' del conjunto de prueba: {:.2f}\nSensibilidad del conjunto de prueba: {:.2f}\n""Precision del conjunto "
      "de prueba: {:.2f}\n""Exactitud del conjunto de prueba: {:.2f} %".format((results[0]),(results[5]),(results[6]),
                                                                       (results[7] * 100)))
print("\n")

"""Las métricas del entreno se guardan dentro del método 'history'. Primero, se definen las variables para usarlas 
posteriormentes para dibujar las gráficas de la 'loss', la sensibilidad y la precisión del entrenamiento y  validación 
de cada iteración."""
loss = neural_network.history['loss']
val_loss = neural_network.history['val_loss']

recall = neural_network.history['recall']
val_recall = neural_network.history['val_recall']

precision = neural_network.history['precision']
val_precision = neural_network.history['val_precision']

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

""" Gráfica de la sensibilidad del entreno y la validación: """
plt.plot(epochs, recall, 'r', label='Sensibilidad del entreno')
plt.plot(epochs, val_recall, 'b--', label='Sensibilidad de la validación')
plt.title('Sensibilidad del entreno y de la validación')
plt.ylabel('Sensibilidad')
plt.xlabel('Epochs')
plt.legend()
plt.figure()

""" Gráfica de la precisión del entreno y la validación: """
plt.plot(epochs, precision, 'r', label='Precisión del entreno')
plt.plot(epochs, val_precision, 'b--', label='Precisión de la validación')
plt.title('Precisión del entreno y de la validación')
plt.ylabel('Precisión')
plt.xlabel('Epochs')
plt.legend()
plt.figure()
plt.show() # Se muestran todas las gráficas

""" Se guarda el modelo en caso de que sea necesario"""
# model.save('model_BRCA1.h5')

""" -------------------------------------------------------------------------------------------------------------------
------------------------------------------- SECCIÓN DE EVALUACIÓN  ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos. """
# @suppress=True: Muestra los números con representación de coma fija
# @predict: Genera predicciones para nuevas entradas
print("\nGenera predicciones para 10 muestras")
print("Etiquetas: ", test_labels[:10])
np.set_printoptions(precision=3, suppress=True)
print("Predicciones:\n", np.round(model.predict(test_image_data[:10])))

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
y_true = test_labels # Etiquetas verdaderas de 'test'
y_pred = np.round(model.predict(test_image_data)) # Predicción de etiquetas de 'test'

matrix = confusion_matrix(y_true, y_pred) # Calcula (pero no dibuja) la matriz de confusión

group_names = ['True Neg','False Pos','False Neg','True Pos'] # Nombres de los grupos
group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()] # Cantidad de casos por grupo

""" @zip: Une las tuplas del nombre de los grupos con la de la cantidad de casos por grupo """
labels = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
plt.show() # Muestra la gráfica de la matriz de confusión

#np.save('test_image', test_image_data)
#np.save('test_labels', test_labels)