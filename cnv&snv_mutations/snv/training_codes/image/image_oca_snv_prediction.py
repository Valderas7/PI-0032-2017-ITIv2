import shelve # datos persistentes
import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
import staintools
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import *
from tensorflow.keras.layers import *
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.model_selection import train_test_split # Se importa la librería para dividir los datos en entreno y test.
from sklearn.preprocessing import MinMaxScaler # Para escalar valores
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix # Para realizar la matriz de confusión

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
    snv = data.get('mutations')

""" Se crea un dataframe para el diccionario de mutaciones SNV y otro para el diccionario de la categoría N del sistema
de estadificación TNM. Posteriormente, se renombran las dos columnas para que todo quede más claro y se fusionan ambos
dataframes. En una columna tendremos el ID del paciente, en otra las distintas mutaciones SNV y en la otra la 
categoría N para dicho paciente. """
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_snv = pd.DataFrame.from_dict(snv.items()); df_snv.rename(columns = {0 : 'ID', 1 : 'SNV'}, inplace = True)

df_list = [df_path_n_stage, df_snv]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

""" Ahora se va a encontrar cuales son los ID de los genes que nos interesa. Para empezar se carga el archivo excel 
donde aparecen todos los genes con mutaciones que interesan estudiar usando 'openpyxl' y creamos dos listas. Una para
los genes SNV y otra para los genes CNV."""
mutations_target = pd.read_excel('/home/avalderas/img_slides/excel_genes_panelOCA/Panel_OCA.xlsx', usecols= 'B:C',
                                 engine= 'openpyxl')

snv = mutations_target.loc[mutations_target['Scope'] != 'CNV', 'Gen']

snv_list = []
for gen_snv in snv:
    if gen_snv not in snv_list:
        snv_list.append(gen_snv)

""" Ahora se recopilan en una lista los distintos IDs de los genes a estudiar. """
id_snv_list = []

key_list = list(dict_genes.keys())
val_list = list(dict_genes.values())

for gen_snv in snv_list:
    position = val_list.index(gen_snv) # Número
    id_gen_snv = (key_list[position]) # Número
    id_snv_list.append(id_gen_snv) # Se añaden todos los IDs en la lista vacía

""" Ahora se hace un bucle sobre la columna de mutaciones SNV del dataframe. Así, se busca en cada mutación de 
cada fila para ver en cuales de estas filas se encuentra el ID de los genes a estudiar. Se almacenan en una lista de 
listas los índices de las filas donde se encuentran los IDs de esos genes, de forma que se tiene una lista para cada gen. """
list_gen_snv = [[] for ID in range(len(id_snv_list))]

for index, id_snv in enumerate (id_snv_list): # Para cada ID del gen SNV de la lista...
    for index_row, row in enumerate (df_all_merge['SNV']): # Para cada fila dentro de la columna 'SNV'...
        for mutation in row: # Para cada mutación dentro de cada fila...
            if mutation[1] == id_snv: # Si el ID de la mutación es el mismo que el ID de la lista de genes...
                list_gen_snv[index].append(index_row) # Se almacena el índice de la fila en la lista de listas

""" Una vez se tienen almacenados los índices de las filas donde se producen las mutaciones SNV y CNV, hay que crear 
distintas columnas para cada uno de los genes objetivo, para asi mostrar la informacion de uno en uno. De esta forma, 
habra una columna distinta para cada gen SNV a estudiar; y tres columnas distintas para cada gen CNV a estudiar 
(amplificacion, delecion y estado normal). Ademas, se recopilan las columnas creadas en listas (una para las 
columnas de mutaciones SNV y otras tres para las columnas de mutaciones CNV). """
columns_list_snv = []

df_all_merge.drop(['SNV'], axis=1, inplace= True)
for gen_snv in snv_list:
    df_all_merge['SNV_' + gen_snv] = 0
    columns_list_snv.append('SNV_' + gen_snv)

""" Una vez han sido creadas las columnas, se añade un '1' en aquellas filas donde el paciente tiene mutación sobre el
gen seleccionado. Se utiliza para ello los índices recogidos anteriormente en las respectivas listas de listas. De esta
forma, iterando sobre la lista de columnas creadas y mediante los distintos indices de cada sublista, se consigue
colocar un '1' en aquella filas donde el paciente tiene la mutacion especificada en el gen especificado. """
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
df_all_merge = df_all_merge[(df_all_merge["path_n_stage"]!='N0') & (df_all_merge["path_n_stage"]!='NX') &
                            (df_all_merge["path_n_stage"]!='N0 (I-)') & (df_all_merge["path_n_stage"]!='N0 (I+)') &
                            (df_all_merge["path_n_stage"]!='N0 (MOL+)')]

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
train_data, test_data = train_test_split(df_all_merge, test_size = 0.20)
train_data, valid_data = train_test_split(train_data, test_size = 0.20)

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
# IMPORTANTE: La anchura no puede ser más alta que la altura.
alto = 100 # 630
ancho = 100 # 1480
canales = 3 # Imágenes a color (RGB) = 3

""" Se leen y se redimensionan posteriormente las imágenes a las dimensiones especificadas arriba: """
train_image_data = [] # Lista con las imágenes redimensionadas
valid_image_data = []
test_image_data = [] # Lista con las imágenes redimensionadas del subconjunto de test

for imagen_train in train_data['img_path']:
    train_image_data.append(cv2.resize(cv2.imread(imagen_train, cv2.IMREAD_COLOR), (ancho, alto),
                                           interpolation=cv2.INTER_CUBIC))

for imagen_valid in valid_data['img_path']:
    valid_image_data.append(cv2.resize(cv2.imread(imagen_valid, cv2.IMREAD_COLOR), (ancho, alto),
                                           interpolation=cv2.INTER_CUBIC))

for imagen_test in test_data['img_path']:
    test_image_data.append(cv2.resize(cv2.imread(imagen_test, cv2.IMREAD_COLOR), (ancho, alto),
                                           interpolation=cv2.INTER_CUBIC))

""" Se convierten las imágenes a un array de numpy para manipularlas con más comodidad y se divide el array entre 255
para escalar los píxeles entre el intervalo (0-1). Como resultado, habrá un array con forma (471, alto, ancho, canales). """
train_image_data = (np.array(train_image_data) / 255.0)
valid_image_data = (np.array(valid_image_data) / 255.0)
test_image_data = (np.array(test_image_data))

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN PROCESAMIENTO DE DATOS -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Una vez ya se tienen las imágenes convertidas en arrays y en el orden establecido por cada paciente, se puede
extraer del dataframe la columna 'SNV', que será la salida de la red:"""
train_labels = train_data.iloc[:,2:-1]
valid_labels = valid_data.iloc[:,2:-1]
test_labels = test_data.iloc[:,2:-1]; test_columns = test_labels.columns.values

""" Los nombres de las distintas clases (columnas) se pasan a una lista para usarlos en un futuro """
classes = test_columns.tolist()

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada, y se recopila la longitud de las imagenes de
entrenamiento y validacion para utilizarlas posteriormente en el entrenamiento: """
del df_all_merge, df_path_n_stage, df_list

train_image_data_len = len(train_image_data)
valid_image_data_len = len(valid_image_data)
batch_dimension = 32

""" Para poder entrenar la red hace falta transformar las tablas en arrays. Para ello se utiliza 'numpy'. Las imágenes 
YA están convertidas en 'arrays' numpy """
train_labels = np.asarray(train_labels).astype('float32')
valid_labels = np.asarray(valid_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL CONVOLUCIONAL ---------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" En esta ocasión, se crea un modelo secuencial para la red neuronal convolucional que será la encargada de procesar
todas las imágenes: """
base_model = keras.applications.ResNet50V2(weights = 'imagenet', input_tensor = Input(shape=(alto, ancho, canales)),
                                              include_top = False)
all_model = base_model.output
all_model = layers.Flatten()(all_model)
all_model = layers.Dense(256)(all_model)
all_model = layers.Dropout(0.5)(all_model)
all_model = layers.Dense(train_labels.shape[1], activation= 'sigmoid')(all_model)
model = keras.models.Model(inputs = base_model.input, outputs = all_model)

""" Se congelan todas las capas convolucionales del modelo base"""
for layer in base_model.layers:
    layer.trainable = False
    
""" Esto se hace para que al hacer el entrenamiento, los pesos de las distintas salidas se balaceen, ya que el conjunto
de datos que se tratan en este problema es muy imbalanceado. """
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = class_weight.compute_sample_weight(class_weight = 'balanced', y = train_labels)
sample_weights_dict = dict(enumerate(sample_weights)) # {0: 2.5243569600427398e-26, 1: 2.5243569600427398e-26, etc}

""" Se realiza data augmentation y definición de la substracción media de píxeles con la que se entrenó la red VGG19.
Como se puede comprobar, solo se aumenta el conjunto de entrenamiento. Los conjuntos de validacion y test solo modifican
la media de pixeles en canal BGR (OpenCV lee las imagenes en formato BGR): """
trainAug = ImageDataGenerator(rescale = 1.0/255, horizontal_flip = True, zoom_range= 0.2, shear_range= 0.2,
                              width_shift_range= 0.2, height_shift_range= 0.2, rotation_range= 20)
valAug = ImageDataGenerator(rescale = 1.0/255)

""" Se instancian las imágenes aumentadas con las variables creadas de imageens y de clases para entrenar estas
instancias posteriormente: """
trainGen = trainAug.flow(x = train_image_data, y = train_labels, batch_size = 32)
valGen = valAug.flow(x = valid_image_data, y = valid_labels, batch_size = 32, shuffle= False)
#testGen = valAug.flow(x = test_image_data, y = test_labels, batch_size = 32, shuffle= False)

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
model.summary()

""" Se implementa un callback: para guardar el mejor modelo que tenga la mayor sensibilidad en la validación. """
checkpoint_path = 'model_snv_image_epoch{epoch:02d}.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = False)

""" Una vez definido el modelo, se entrena: """
model.fit(trainGen, epochs = 10, verbose = 1, validation_data = valGen,
          steps_per_epoch = (train_image_data_len / batch_dimension),
          validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez el modelo ya ha sido entrenado, se resetean los generadores de data augmentation de los conjuntos de 
entrenamiento y validacion y se descongelan algunas capas convolucionales del modelo base de la red para reeentrenar
todo el modelo de principio a fin ('fine tuning'). Este es un último paso opcional que puede dar grandes mejoras o un 
rápido sobreentrenamiento y que solo debe ser realizado después de entrenar el modelo con las capas congeladas. 
Para ello, primero se descongela el modelo base."""
trainGen.reset()
valGen.reset()

for layer in base_model.layers[15:]:
    layer.trainable = True

""" Es importante recompilar el modelo después de hacer cualquier cambio al atributo 'trainable', para que los cambios
se tomen en cuenta (y se utiliza un 'learning rate' muy pequeño): """
model.compile(optimizer = keras.optimizers.Adam (learning_rate = 1e-5),
              loss = 'binary_crossentropy',
              metrics = metrics)
model.summary()

""" Se entrena el modelo de principio a fin: """
""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
neural_network = model.fit(trainGen, epochs = 100, verbose = 1, validation_data = valGen,
                           steps_per_epoch = (train_image_data_len / batch_dimension),
                           validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data, test_labels, verbose = 0)
print("\n'Loss' del conjunto de prueba: {:.2f}\n""Sensibilidad del conjunto de prueba: {:.2f}\n" 
      "Precisión del conjunto de prueba: {:.2f}\n""Especifidad del conjunto de prueba: {:.2f} \n"
      "Exactitud del conjunto de prueba: {:.2f} %\n" 
      "El AUC-ROC del conjunto de prueba es de: {:.2f}".format(results[0], results[5], results[6],
                                                               results[3]/(results[3]+results[2]), results[7] * 100,
                                                               results[8]))

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

""" -------------------------------------------------------------------------------------------------------------------
------------------------------------------- SECCIÓN DE EVALUACIÓN  ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos. """
# @suppress=True: Muestra los números con representación de coma fija
# @predict: Genera predicciones para nuevas entradas
print("\nGenera predicciones para la primera muestra:")
print("Clases para la primera muestra: ", test_labels[:1]); print("\n")
np.set_printoptions(precision=3, suppress=True)
print("Predicciones para la primera muestra:\n", model.predict(test_image_data[:1])); print("\n")
proba = model.predict(test_image_data[:1])[0] # Muestra las predicciones pero en una sola dimension
idxs = np.argsort(proba)[::-1][:1] # Muestra los dos indices mas altos de las predicciones

for (i, j) in enumerate(idxs):
    label = "La mutacion SNV más probable de esta imagen es del gen {}: {:.2f}%".format(classes[j][4:], proba[j] * 100)
    print(label)

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
y_true = test_labels # Etiquetas verdaderas de 'test'
y_pred = np.round(model.predict(test_image_data)) # Predicción de etiquetas de 'test'
y_pred_prob = model.predict(test_image_data).ravel()

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
    print("Matriz de confusion para el gen {}:".format(label))
    print(matrix)
    if matrix.shape == (2,2):
        """ @zip: Une las tuplas del nombre de los grupos con la de la cantidad de casos por grupo """
        group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()]  # Cantidad de casos por grupo
        true_neg_pos_neg = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
        true_neg_pos_neg = np.asarray(true_neg_pos_neg).reshape(2, 2)
        #sns.heatmap(matrix, annot=true_neg_pos_neg, fmt='', cmap='Blues')
        #plt.title(label)
        #plt.show()
        #print("\n")

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

micro_roc_auc_ovr = roc_auc_score(test_labels, y_pred_prob, multi_class="ovr",
                                     average="micro")
micro_pr_auc_ovr = average_precision_score(test_labels, y_pred_prob, average="micro")

print("Puntuación AUC-ROC: {:.2f} (micro-promedio)\n".format(micro_roc_auc_ovr))
print("Puntuación AUC-PR: {:.2f} (micro-promedio)\n".format(micro_pr_auc_ovr))

""" Una vez calculadas las dos puntuaciones, se dibuja la curva micro-promedio. Esto es mejor que dibujar una curva para 
cada una de las clases que hay en el problema. """
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(classes)):
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

#np.save('test_image', test_image_data)
#np.save('test_labels', test_labels)