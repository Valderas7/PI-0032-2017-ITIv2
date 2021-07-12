import shelve # datos persistentes
import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
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
#filename = 'C:\\Users\\valde\Desktop\Datos_repositorio\cbioportal\data/brca_tcga_pan_can_atlas_2018.out'

""" Almacenamos en una variable los diccionarios: """
with shelve.open(filename) as data:
    dict_genes = data.get('dict_genes')
    path_n_stage = data.get('path_n_stage')
    path_m_stage = data.get('path_m_stage')

""" Se crean dataframes individuales para cada uno de los diccionarios almacenados en cada variable de entrada y se
renombran las columnas para que todo quede más claro. Además, se crea una lista con todos los dataframes para
posteriormente unirlos todos juntos. """
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_path_m_stage = pd.DataFrame.from_dict(path_m_stage.items()); df_path_m_stage.rename(columns = {0 : 'ID', 1 : 'path_m_stage'}, inplace = True)

df_list = [df_path_n_stage, df_path_m_stage]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

""" En este caso, el número de muestras de imágenes y de datos deben ser iguales. Las imágenes de las que se disponen se 
enmarcan según el sistema de estadificación TNM como N1A, N1, N2A, N2, N3A, N1MI, N1B, N3, NX, N3B, N1C o N3C según la
categoría N (extensión de cáncer que se ha diseminado a los ganglios linfáticos) de dicho sistema de estadificación.
Por tanto, en los datos tabulares tendremos que quedarnos solo con los casos donde los pacientes tengan esos valores
de la categoría 'N' y habrá que usar, por tanto, una imagen para cada paciente, para que no haya errores al repartir
los subconjuntos de datos. """
# 552 filas resultantes, como en cBioPortal:
df_all_merge = df_all_merge[(df_all_merge["path_n_stage"]!='N0') & (df_all_merge["path_n_stage"]!='NX') &
                            (df_all_merge["path_n_stage"]!='N0 (I-)') & (df_all_merge["path_n_stage"]!='N0 (I+)') &
                            (df_all_merge["path_n_stage"]!='N0 (MOL+)') & (df_all_merge["path_m_stage"]!='MX')]

""" Al realizar un análisis de los datos de entrada se ha visto un único valor incorrecto en la columna
'cancer_type_detailed'. Por ello se sustituye dicho valor por 'Breast Invasive Carcinoma (NOS)'. También se ha apreciado
un único valor en 'tumor_type', por lo que también se realiza un cambio de valor en dicho valor atípico. Además, se 
convierten las columnas categóricas binarias a valores de '0' y '1', para no aumentar el número de columnas: """
df_all_merge.loc[df_all_merge.path_m_stage == "CM0 (I+)", "path_m_stage"] = 'M0'
df_all_merge.loc[df_all_merge.path_m_stage == "M0", "path_m_stage"] = 0; df_all_merge.loc[df_all_merge.path_m_stage == "M1", "path_m_stage"] = 1

""" Ahora, antes de transformar las variables categóricas en numéricas, se eliminan las filas donde haya datos nulos
para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
train_data, test_data = train_test_split(df_all_merge, test_size = 0.20, stratify = df_all_merge['path_m_stage'],
                                         random_state = 42)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------------- SECCIÓN IMÁGENES -------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Directorios de imágenes con cáncer y sin cáncer: """
image_dir = '/home/avalderas/img_slides/img_lotes'
#image_dir = 'C:\\Users\\valde\Desktop\Datos_repositorio\img_slides\img_lotes'

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
alto = int(100) # Eje Y: 630. Nº de filas
ancho = int(100) # Eje X: 1480. Nº de columnas
canales = 3 # Imágenes a color (RGB) = 3

mitad_alto = int(alto/2)
mitad_ancho = int(ancho/2)

""" Se leen y se redimensionan posteriormente las imágenes de ambos subconjuntos a las dimensiones especificadas arriba
y se añaden a una lista: """
pre_train_image_data = [] # Lista con las imágenes redimensionadas del subconjunto de entrenamiento
test_image_data = [] # Lista con las imágenes redimensionadas del subconjunto de test

for imagen_train in train_data['img_path']:
    pre_train_image_data.append(cv2.resize(cv2.imread(imagen_train,cv2.IMREAD_COLOR),(ancho,alto),
                                           interpolation=cv2.INTER_CUBIC))

for imagen_test in test_data['img_path']:
    test_image_data.append(cv2.resize(cv2.imread(imagen_test,cv2.IMREAD_COLOR),(ancho,alto),
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

""" Se convierten las imágenes a un array de numpy para poderlas introducir posteriormente en el modelo de red. Además,
se divide todo el array de imágenes entre 255 para escalar los píxeles en el intervalo (0-1). Como resultado, habrá un 
array con forma (X, alto, ancho, canales). """
train_image_data = (np.array(train_image_data) / 255.0)
test_image_data = (np.array(test_image_data) / 255.0)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN PROCESAMIENTO DE DATOS -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Una vez se tienen hechos los recortes de imágenes, se procede a replicar las filas de ambos subconjuntos de datos
para que el número de imágenes utilizadas y el número de filas del marco de datos sea el mismo: """
train_data = pd.DataFrame(np.repeat(train_data.values, 9, axis=0), columns=train_data.columns)

""" Una vez ya se tienen las imágenes convertidas en arrays de numpy, se puede eliminar de los dos subconjuntos tanto la
columna 'ID' como la columna 'path_img' que no son útiles para la red MLP: """
#@inplace = True para que devuelva el resultado en la misma variable
train_data.drop(['ID'], axis=1, inplace= True)
train_data.drop(['img_path'], axis=1, inplace= True)

test_data.drop(['ID'], axis=1, inplace= True)
test_data.drop(['img_path'], axis=1, inplace= True)

""" Se extrae la columna 'path_m_stage' del dataframe de ambos subconjuntos, puesto que ésta es la salida del modelo que 
se va a entrenar."""
train_labels = train_data.pop('path_m_stage')
test_labels = test_data.pop('path_m_stage')

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada: """
del df_all_merge, df_path_n_stage, df_list

""" Para poder entrenar la red hace falta transformar los dataframes de entrenamiento y test en arrays de numpy, así 
como también la columna de salida de ambos subconjuntos (las imágenes YA fueron convertidas anteriormente, por lo que no
hace falta transformarlas de nuevo). """
train_labels = np.asarray(train_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (MLP + CNN) -----------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Se define la red neuronal convolucional: """
cnn_model = keras.applications.EfficientNetB7(weights='imagenet', input_shape=(alto, ancho, canales),
                                              include_top=False)
cnn_model.trainable = False

inputs = keras.Input(shape=(alto, ancho, canales))
x = cnn_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(1, activation= 'sigmoid')(x)
model = keras.models.Model(inputs = inputs, outputs = x)
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
              optimizer = keras.optimizers.Adam(learning_rate = 0.001),
              metrics = metrics)

""" Se implementa un callback: para guardar el mejor modelo que tenga la mayor sensibilidad en la validación. """
checkpoint_path = 'model_image_distant_metastasis_status_epoch{epoch:02d}.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = False)

""" Esto se hace para que al hacer el entrenamiento, los pesos de las distintas salidas se balaceen, ya que el conjunto
de datos que se tratan en este problema es muy imbalanceado. """
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels),
                                                  y = train_labels)
class_weight_dict = dict(enumerate(class_weights))

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
neural_network = model.fit(x = train_image_data,  # Datos de entrada.
                           y = train_labels,  # Datos objetivos.
                           epochs = 3,
                           verbose = 1,
                           batch_size= 32,
                           class_weight= class_weight_dict,
                           #callbacks= mcp_save,
                           validation_split = 0.2) # Datos de validación.

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
#model = keras.models.load_model('model_cnv_pik3ca_epoch{epoch:02d}-recall{val_recall:.2f}-precision{val_precision:.2f}.h5')
results = model.evaluate(test_image_data, test_labels, verbose = 0)
print("\n'Loss' del conjunto de prueba: {:.2f}\n""Sensibilidad del conjunto de prueba: {:.2f}\n" 
      "Precisión del conjunto de prueba: {:.2f}\n""Exactitud del conjunto de prueba: {:.2f} %\n"
      "El AUC ROC del conjunto de prueba es de: {:.2f}".format(results[0],results[5],results[6],results[7] * 100,
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
# @suppress=True: Muestra los números con representación de coma fija
# @predict: Genera predicciones para nuevas entradas
print("\nGenera predicciones para 10 muestras")
print("Clase de las salidas: ", test_labels[:10])
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

""" Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases.
Para implementarlas, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva: """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve

y_pred_prob = model.predict(test_image_data).ravel()
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
auc_roc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC-AUC curve')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision, recall, threshold = precision_recall_curve(y_true, y_pred_prob)
auc_pr = auc(recall, precision)

plt.figure(2)
plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-AUC curve')
plt.legend(loc = 'best')
plt.show()

#np.save('test_image', test_image_data)
#np.save('test_labels', test_labels)