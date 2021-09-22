import shelve # datos persistentes
import pandas as pd
import numpy as np
import itertools
import staintools
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
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
alto = int(100) # Eje Y: 630. Nº de filas
ancho = int(100) # Eje X: 1480. Nº de columnas
canales = 3 # Imágenes a color (RGB) = 3

""" Se establece la primera imagen como la imagen objetivo respecto a la que normalizar el color, se estandariza también
el brillo para mejorar el cálculo y depués de normalizar el color, se redimensionan las imágenes, añadiéndolas
posteriormente a su respectiva lista del subconjunto de datos"""
# @StainNormalizer: Instancia para normalizar el color de la imagen mediante el metodo de normalizacion especificado
train_image_data = [] # Lista con las imágenes redimensionadas del subconjunto de entrenamiento
valid_image_data = [] # Lista con las imágenes redimensionadas del subconjunto de validación
test_image_data = [] # Lista con las imágenes redimensionadas del subconjunto de test

normalizer = staintools.StainNormalizer(method='vahadane')
#augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)

""" Filtro de detección de bordes (enfocar) """
kernel = np.array([[-1.0, -1.0, -1.0],
                   [-1.0, 5.0, -1.0],
                   [-1.0, -1.0, -1.0]])

for index_normal_train, imagen_train in enumerate(train_data['img_path']):
    if index_normal_train == 0:
        target = staintools.read_image(imagen_train)
        target = staintools.LuminosityStandardizer.standardize(target)
        normalizer.fit(target)

    img_train = staintools.read_image(imagen_train)

    normal_image_train = staintools.LuminosityStandardizer.standardize(img_train)
    normal_image_train = normalizer.transform(normal_image_train)

    img_train_norm_resize = cv2.resize(normal_image_train, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    #img_train_norm_resize = cv2.filter2D(img_train_norm_resize, -1, kernel)
    #img_train_norm_resize = cv2.cvtColor(img_train_norm_resize, cv2.COLOR_RGB2HSV_FULL)
    train_image_data.append(img_train_norm_resize)

for imagen_valid in valid_data['img_path']:
    img_valid = staintools.read_image(imagen_valid)
    normal_image_valid = staintools.LuminosityStandardizer.standardize(img_valid)
    normal_image_valid = normalizer.transform(normal_image_valid)
    img_valid_norm_resize = cv2.resize(normal_image_valid, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    #img_valid_norm_resize = cv2.filter2D(img_valid_norm_resize, -1, kernel)
    #img_valid_norm_resize = cv2.cvtColor(img_valid_norm_resize, cv2.COLOR_RGB2HSV_FULL)
    valid_image_data.append(img_valid_norm_resize)

for imagen_test in test_data['img_path']:
    img_test = staintools.read_image(imagen_test)
    normal_image_test = staintools.LuminosityStandardizer.standardize(img_test)
    normal_image_test = normalizer.transform(normal_image_test)
    img_test_norm_resize = cv2.resize(normal_image_test, (ancho, alto), interpolation=cv2.INTER_CUBIC)
    #img_test_norm_resize = cv2.filter2D(img_test_norm_resize, -1, kernel)
    #img_test_norm_resize = cv2.cvtColor(img_test_norm_resize, cv2.COLOR_RGB2HSV_FULL)

    test_image_data.append(img_test_norm_resize)

""" Se convierten las imágenes a un array de numpy para poderlas introducir posteriormente en el modelo de red. Además,
se divide todo el array de imágenes entre 255 para escalar los píxeles en el intervalo (0-1). Como resultado, habrá un 
array con forma (X, alto, ancho, canales). """
train_image_data = np.array(train_image_data)
valid_image_data = np.array(valid_image_data)
test_image_data = (np.array(test_image_data) / 255.0)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN PROCESAMIENTO DE DATOS -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Una vez ya se tienen las imágenes convertidas en arrays de numpy, se puede eliminar de los dos subconjuntos tanto la
columna 'ID' como la columna 'path_img' que no son útiles para la red MLP: """
train_data = train_data.drop(['ID'], axis=1)
train_data = train_data.drop(['img_path'], axis=1)

valid_data = valid_data.drop(['ID'], axis=1)
valid_data = valid_data.drop(['img_path'], axis=1)

test_data = test_data.drop(['ID'], axis=1)
test_data = test_data.drop(['img_path'], axis=1)

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

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada, y se recopila la longitud de las imagenes de
entrenamiento y validacion para utilizarlas posteriormente en el entrenamiento: """
del df_all_merge, df_path_n_stage, df_list, train_data

train_image_data_len = len(train_image_data)
valid_image_data_len = len(valid_image_data)
batch_dimension = 32

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (CNN) -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" En esta ocasión, se crea un modelo secuencial para la red neuronal convolucional que será la encargada de procesar
todas las imágenes: """
base_model = keras.applications.VGG16(weights = 'imagenet', input_tensor = Input(shape=(alto, ancho, canales)),
                                              include_top = False)
all_model = base_model.output
all_model = layers.Flatten()(all_model)
all_model = layers.Dense(256)(all_model)
all_model = layers.Dropout(0.5)(all_model)
all_model = layers.Dense(len(lb.classes_), activation= 'softmax')(all_model)
model = keras.models.Model(inputs = base_model.input, outputs = all_model)

""" Se congelan todas las capas convolucionales del modelo base"""
for layer in base_model.layers:
    layer.trainable = False

""" Se realiza data augmentation y definición de la substracción media de píxeles con la que se entrenó la red VGG19.
Como se puede comprobar, solo se aumenta el conjunto de entrenamiento. Los conjuntos de validacion y test solo modifican
la media de pixeles en canal BGR (OpenCV lee las imagenes en formato BGR): """
trainAug = ImageDataGenerator(rescale = 1.0/255, horizontal_flip = True, vertical_flip = True, zoom_range= 0.2,
                              shear_range= 0.2, width_shift_range= 0.2, height_shift_range= 0.2, rotation_range= 20)
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

model.compile(loss = 'categorical_crossentropy', # Esta función de loss suele usarse para clasificación binaria.
              optimizer = keras.optimizers.Adam(learning_rate = 0.001),
              metrics = metrics)
model.summary()

""" Se implementa un callback: para guardar el mejor modelo que tenga la mayor sensibilidad en la validación. """
checkpoint_path = '../../training_codes/image/model_image_distant_metastasis_prediction.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = False)

""" Esto se hace para que al hacer el entrenamiento, los pesos de las distintas salidas se balaceen, ya que el conjunto
de datos que se tratan en este problema es muy imbalanceado. """
from sklearn.utils.class_weight import compute_class_weight

y_integers = np.argmax(train_labels, axis=1)
class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_integers), y = y_integers)
d_class_weights = dict(enumerate(class_weights)) # {0: 1.4780, 1: 2.055238, 2: 0.40186, 3: 0.85... etc}

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
model.fit(trainGen, epochs = 100, verbose = 1, steps_per_epoch = (train_image_data_len / batch_dimension),
          class_weight = d_class_weights, validation_data = valGen,
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
se tomen en cuenta """
model.compile(loss = 'categorical_crossentropy', # Esta función de loss suele usarse para clasificación binaria.
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)
model.summary()

""" Una vez descongeladas las capas convolucionales seleccionadas y compilado de nuevo el modelo, se entrena otra vez. """
neural_network = model.fit(trainGen, epochs = 500, verbose = 1, validation_data = valGen,
                           steps_per_epoch = (train_image_data_len / batch_dimension), class_weight = d_class_weights,
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

matrix = confusion_matrix(y_true, y_pred) # Calcula (pero no dibuja) la matriz de confusión
matrix_classes = lb.classes_

""" Función para mostrar por pantalla la matriz de confusión multiclase con todas las clases de subtipos moleculares """
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión', cmap = plt.cm.Reds):
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
plot_confusion_matrix(matrix, classes = matrix_classes, title='Matriz de confusión multiclase')
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

micro_roc_auc_ovr = roc_auc_score(test_labels, y_pred_prob, multi_class="ovr",
                                     average="micro")
weighted_roc_auc_ovr = roc_auc_score(test_labels, y_pred_prob, multi_class="ovr",
                                     average="weighted")
micro_pr_auc_ovr = average_precision_score(test_labels, y_pred_prob, average="micro")
weighted_pr_auc_ovr = average_precision_score(test_labels, y_pred_prob, average="weighted")

print("Puntuaciones AUC-ROC:\n{:.2f} (micro-promedio)\n{:.2f} (promedio ponderado)\n".format(micro_roc_auc_ovr,
                                                                                             weighted_roc_auc_ovr))
print("Puntuaciones AUC-PR:\n{:.2f} (micro-promedio)\n{:.2f} (promedio ponderado)".format(micro_pr_auc_ovr,
                                                                                          weighted_pr_auc_ovr))

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

#np.save('test_image', test_image_data)
#np.save('test_labels', test_labels)