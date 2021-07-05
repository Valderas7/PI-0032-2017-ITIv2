import shelve # datos persistentes
import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import glob
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input # Para instanciar tensores de Keras
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.model_selection import train_test_split # Se importa la librería para dividir los datos en entreno y test.
from sklearn.preprocessing import MinMaxScaler # Para escalar valores
from sklearn.metrics import confusion_matrix # Para realizar la matriz de confusión

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN DATOS TABULARES ------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""

""" - Datos de entrada: Age, cancer_type, cancer_type_detailed, neoadjuvant, path_m_stage, path_n_stage, path_t_stage, 
    stage, subtype. (prior_diagnosis hay que preguntar)
    - Salida binaria: Presenta mutación o no en el gen 'X' (CNV). """
list_to_read = ['CNV_oncomine', 'age', 'all_oncomine', 'mutations_oncomine', 'cancer_type', 'cancer_type_detailed',
                'dfs_months', 'dfs_status', 'dict_genes', 'dss_months', 'dss_status', 'ethnicity',
                'full_length_oncomine', 'fusions_oncomine', 'muted_genes', 'CNA_genes', 'hotspot_oncomine', 'mutations',
                'CNAs', 'neoadjuvant', 'os_months', 'os_status', 'path_m_stage', 'path_n_stage', 'path_t_stage', 'sex',
                'stage', 'subtype', 'tumor_type', 'new_tumor', 'person_neoplasm_status', 'prior_diagnosis',
                'pfs_months', 'pfs_status', 'radiation_therapy']

filename = '/home/avalderas/img_slides/data/brca_tcga_pan_can_atlas_2018.out'
#filename = 'C:\\Users\\valde\Desktop\Datos_repositorio\cbioportal\data/brca_tcga_pan_can_atlas_2018.out'

""" Se almacena en cada variable un diccionario. """
with shelve.open(filename) as data:
    dict_genes = data.get('dict_genes')
    age = data.get('age')
    cancer_type_detailed = data.get('cancer_type_detailed')
    neoadjuvant = data.get('neoadjuvant') # 1 valor nulo
    os_months = data.get('os_months')
    os_status = data.get('os_status')
    path_m_stage = data.get('path_m_stage')
    path_n_stage = data.get('path_n_stage')
    path_t_stage = data.get('path_t_stage')
    stage = data.get('stage')
    subtype = data.get('subtype')
    tumor_type = data.get('tumor_type')
    new_tumor = data.get('new_tumor') # ~200 valores nulos
    prior_diagnosis = data.get('prior_diagnosis') # 1 valor nulo
    pfs_months = data.get('pfs_months') # 2 valores nulos
    pfs_status = data.get('pfs_status') # 1 valor nulo
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
df_os_months = pd.DataFrame.from_dict(os_months.items()); df_os_months.rename(columns = {0 : 'ID', 1 : 'os_months'}, inplace = True)
df_os_status = pd.DataFrame.from_dict(os_status.items()); df_os_status.rename(columns = {0 : 'ID', 1 : 'os_status'}, inplace = True)
df_path_m_stage = pd.DataFrame.from_dict(path_m_stage.items()); df_path_m_stage.rename(columns = {0 : 'ID', 1 : 'path_m_stage'}, inplace = True)
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_path_t_stage = pd.DataFrame.from_dict(path_t_stage.items()); df_path_t_stage.rename(columns = {0 : 'ID', 1 : 'path_t_stage'}, inplace = True)
df_stage = pd.DataFrame.from_dict(stage.items()); df_stage.rename(columns = {0 : 'ID', 1 : 'stage'}, inplace = True)
df_subtype = pd.DataFrame.from_dict(subtype.items()); df_subtype.rename(columns = {0 : 'ID', 1 : 'subtype'}, inplace = True)
df_tumor_type = pd.DataFrame.from_dict(tumor_type.items()); df_tumor_type.rename(columns = {0 : 'ID', 1 : 'tumor_type'}, inplace = True)
df_new_tumor = pd.DataFrame.from_dict(new_tumor.items()); df_new_tumor.rename(columns = {0 : 'ID', 1 : 'new_tumor'}, inplace = True)
df_prior_diagnosis = pd.DataFrame.from_dict(prior_diagnosis.items()); df_prior_diagnosis.rename(columns = {0 : 'ID', 1 : 'prior_diagnosis'}, inplace = True)

df_cnv = pd.DataFrame.from_dict(cnv.items()); df_cnv.rename(columns = {0 : 'ID', 1 : 'CNV'}, inplace = True)

df_list = [df_age, df_cancer_type_detailed, df_neoadjuvant, df_path_m_stage, df_path_n_stage, df_path_t_stage, df_stage,
           df_subtype, df_tumor_type, df_prior_diagnosis, df_os_status, df_cnv]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

""" Ahora vamos a encontrar cual es el ID del gen que se quiere predecir. Para ello se crean dos variables para
crear una lista de claves y otra de los valores del diccionario de genes. Se extrae el índice del gen en la lista de
valores y posteriormente se usa ese índice para buscar con qué clave (ID) se corresponde en la lista de claves. """
key_list = list(dict_genes.keys())
val_list = list(dict_genes.values())

position = val_list.index('BRCA1') # Número AQUÍ ESPECIFICAMOS EL GEN CUYA MUTACIÓN CNV SE QUIERE PREDECIR
id_gen = (key_list[position]) # Número

""" Se hace un bucle sobre la columna de mutaciones del dataframe. Así, se busca en cada mutación de cada fila para ver
en que filas se puede encontrar el ID del gen que se quiere predecir. Se almacenan en una lista los índices de las filas
donde se encuentra ese ID. """
list_gen = []

""" Se crea esta lista de los pacientes que tienen mutación 'CNV' de estos genes porque hay un fallo en el diccionario 
de mutaciones 'CNV' y no identifica sus mutaciones. Por tanto, se ha recopilado manualmente los 'IDs' de los pacientes
que tienen mutaciones en el gen (gracias a cBioPortal) para poner un '1' en la columna 'CNV' de esos 'IDs'. """
brca1_list = ['TCGA-A2-A0EO', 'TCGA-A7-A13D', 'TCGA-A8-A09G', 'TCGA-AC-A2FB', 'TCGA-AN-A04C', 'TCGA-AR-A24H',
              'TCGA-B6-A0IG', 'TCGA-B6-A0IN', 'TCGA-BH-A0AW', 'TCGA-BH-A0C0', 'TCGA-BH-A42T', 'TCGA-C8-A12L',
              'TCGA-C8-A9FZ', 'TCGA-E2-A105', 'TCGA-E2-A1L7', 'TCGA-E9-A1RI', 'TCGA-EW-A1OX', 'TCGA-LD-A9QF']

brca2_list = ['TCGA-A2-A04T', 'TCGA-A7-A0CE', 'TCGA-A8-A06R', 'TCGA-A8-A08I', 'TCGA-A8-A09V', 'TCGA-A8-A0AB',
              'TCGA-AN-A04D', 'TCGA-AN-A0AS', 'TCGA-AR-A24H', 'TCGA-B6-A0IQ', 'TCGA-BH-A0GZ', 'TCGA-BH-A1EV',
              'TCGA-D8-A147', 'TCGA-D8-A1JB', 'TCGA-D8-A1JD', 'TCGA-D8-A1Y2', 'TCGA-E2-A14T', 'TCGA-E2-A1LG',
              'TCGA-EW-A1OX', 'TCGA-EW-A1P7', 'TCGA-PE-A5DC', 'TCGA-S3-AA10']

cdkn1b_list = ['TCGA-A1-A0SK', 'TCGA-A1-A0SP', 'TCGA-A2-A04T', 'TCGA-A2-A04U', 'TCGA-A2-A3XT', 'TCGA-A7-A4SD',
              'TCGA-A7-A6VW', 'TCGA-A8-A06R', 'TCGA-AC-A2FM', 'TCGA-AN-A0AJ', 'TCGA-AN-A0FJ', 'TCGA-AQ-A54N',
              'TCGA-AR-A24M', 'TCGA-C8-A12L', 'TCGA-C8-A1HJ', 'TCGA-E9-A22G', 'TCGA-LL-A8F5', 'TCGA-OL-A5RU']

if id_gen == 672: # BRCA1
    for patient_brca1 in brca1_list:
        for index_brca1, row_brca1 in enumerate(df_all_merge['ID']):
            if patient_brca1 == row_brca1:
                list_gen.append(index_brca1)
elif id_gen == 675: # BRCA2
    for patient_brca2 in brca2_list:
        for index_brca2, row_brca2 in enumerate(df_all_merge['ID']):
            if patient_brca2 == row_brca2:
                list_gen.append(index_brca2)
elif id_gen == 1027: # CDKN1B
    for patient_cdkn1b in cdkn1b_list:
        for index_cdkn1b, row_cdkn1b in enumerate(df_all_merge['ID']):
            if patient_cdkn1b == row_cdkn1b:
                list_gen.append(index_cdkn1b)
else:
    for index, row in enumerate (df_all_merge['CNV']): # Para cada fila...
        for mutation in row: # Para cada mutación de cada fila...
            if mutation[1] == id_gen: # Si el segundo elemento de la lista es el mismo número que identifica al gen...
                list_gen.append(index)

""" Una vez se tienen almacenados los índices de las filas donde se produce esa mutación, como la salida de la red será
binaria, se transforman todos los valores de la columna 'SNV' a '0' (no hay mutación del gen específico). Y una 
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

""" Al realizar un análisis de los datos de entrada se ha visto un único valor incorrecto en la columna
'cancer_type_detailed'. Por ello se sustituye dicho valor por 'Breast Invasive Carcinoma (NOS)'. También se ha apreciado
un único valor en 'tumor_type', por lo que también se realiza un cambio de valor en dicho valor atípico. Además, se 
convierten las columnas categóricas binarias a valores de '0' y '1', para no aumentar el número de columnas: """
df_all_merge.loc[df_all_merge.cancer_type_detailed == "Invasive Breast Carcinoma", "cancer_type_detailed"] = "Breast Invasive Carcinoma (NOS)"
df_all_merge.loc[df_all_merge.tumor_type == "Infiltrating Carcinoma (NOS)", "tumor_type"] = "Mixed Histology (NOS)"
df_all_merge.loc[df_all_merge.tumor_type == "Breast Invasive Carcinoma", "tumor_type"] = "Infiltrating Ductal Carcinoma"
df_all_merge.loc[df_all_merge.neoadjuvant == "No", "neoadjuvant"] = 0; df_all_merge.loc[df_all_merge.neoadjuvant == "Yes", "neoadjuvant"] = 1
df_all_merge.loc[df_all_merge.prior_diagnosis == "No", "prior_diagnosis"] = 0; df_all_merge.loc[df_all_merge.prior_diagnosis == "Yes", "prior_diagnosis"] = 1
df_all_merge.loc[df_all_merge.os_status == "0:LIVING", "os_status"] = 0; df_all_merge.loc[df_all_merge.os_status == "1:DECEASED", "os_status"] = 1

""" Ahora, antes de transformar las variables categóricas en numéricas, se eliminan las filas donde haya datos nulos
para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Una vez la tabla tiene las columnas deseadas se procede a codificar las columnas categóricas del dataframe a valores
numéricos mediante la técnica del 'One Hot Encoding'. Más adelante se escalarán las columnas numéricas continuas, pero
ahora se realiza esta técnica antes de hacer la repartición de subconjuntos para que no haya problemas con las columnas. """
#@ get_dummies: Aplica técnica de 'One Hot Encoding', creando columnas binarias para las columnas seleccionadas
df_all_merge = pd.get_dummies(df_all_merge, columns=["cancer_type_detailed","path_m_stage","path_n_stage",
                                                     "path_t_stage", "stage", "subtype","tumor_type"])

""" Se dividen los datos tabulares y la columna de salida en conjuntos de entrenamiento y test. Se hace a estas alturas 
para prevenir el posterior sobreentrenamiento y evitar que filas con los datos del mismo paciente se presenten en ambos
subconjuntos de entrenamiento y test simultáneamente, ya que al añadir las imágenes, las filas se multiplicarán: """
# @train_test_split: Hace la repartición de los subconjuntos de entrenamiento y test
# @random_state: Se establece una semilla para que en cada ejecución la repartición sea la misma, aunque esté barajada
# @stratify: Mantiene la proporción de valores de la variable especificada entre ambos subconjuntos de datos
# IMPORTANTE: Los datos y las imágenes tienen que tener el mismo número de muestras, si no, hay error.
train_tabular_data, test_tabular_data = train_test_split(df_all_merge, test_size = 0.20, stratify= df_all_merge['CNV'],
                                                         random_state = 42)

train_tabular_data, valid_tabular_data = train_test_split(train_tabular_data, test_size = 0.20,
                                                          stratify= train_tabular_data['CNV'], random_state = 42)

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
Por último, ambos subconjuntos de datos se unen con la serie mediante la columna 'ID'. De esta forma, cada paciente 
replicará sus filas el número de veces que tenga una imagen distinta, es decir, que si un paciente tiene tres imágenes, 
la fila de datos de ese paciente se presenta 3 veces, teniendo en cada una de ellas una ruta de imagen distinta. Y como
los subconjuntos de datos ya han sido creados, no van a coincidir filas entre ambos subconjuntos puesto que en cada uno 
de ellos hay pacientes distintos: """
series_img = pd.Series(cancer_dir)
series_img.index = series_img.str.extract(fr"({'|'.join(df_all_merge['ID'])})", expand=False)

train_tabular_data = train_tabular_data.join(series_img.rename('img_path'), on='ID')
valid_tabular_data = valid_tabular_data.join(series_img.rename('img_path'), on='ID')
test_tabular_data = test_tabular_data.join(series_img.rename('img_path'), on='ID')

""" Hay valores nulos, por lo que se ha optado por eliminar esas filas en ambos subconjuntos para que se pueda entrenar 
posteriormente el modelo. Aparte de eso, se ordena el dataframe según los valores de la columna 'ID': """
# 1510 filas resultantes entre ambos subconjuntos, como en la versión anterior del programa:
train_tabular_data.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.
train_tabular_data = train_tabular_data.sort_index()

valid_tabular_data.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.
test_tabular_data.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Una vez se tienen todas las imágenes y quitados los valores nulos, tambiés es necesario deshacernos de aquellas 
imágenes que son intraoperatorias. Para ello se toma como referencia el archivo 'Pacientes_MGR' para eliminar las filas 
de aquellas imágenes que no nos van a servir. Así, se recogen los índices de aquellas filas de ambos subconjuntos donde 
aparecen esas imágenes, y una vez encontrados esos índices, se eliminan"""
# 1483 filas resultantes entre ambos subconjuntos, como en la versión anterior del programa:
remove_img_list = ['TCGA-A2-A0EW', 'TCGA-E2-A153', 'TCGA-E2-A15A', 'TCGA-E2-A15E', 'TCGA-E9-A1N4', 'TCGA-E9-A1N5',
                   'TCGA-E9-A1N6', 'TCGA-E9-A1NC', 'TCGA-E9-A1ND', 'TCGA-E9-A1NE', 'TCGA-E9-A1NH', 'TCGA-PL-A8LX',
                   'TCGA-PL-A8LZ']

for id_img in remove_img_list:
    index_train = train_tabular_data.loc[df_all_merge['ID'] == id_img].index
    index_valid = valid_tabular_data.loc[df_all_merge['ID'] == id_img].index
    index_test = test_tabular_data.loc[df_all_merge['ID'] == id_img].index
    train_tabular_data.drop(index_train, inplace=True)
    valid_tabular_data.drop(index_valid, inplace=True)
    test_tabular_data.drop(index_test, inplace=True)

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
valid_image_data = [] # Lista con las imágenes redimensionadas del subconjunto de validación
test_image_data = [] # Lista con las imágenes redimensionadas del subconjunto de test

for imagen_train in train_tabular_data['img_path']:
    pre_train_image_data.append(cv2.resize(cv2.imread(imagen_train,cv2.IMREAD_COLOR),(ancho,alto),
                                           interpolation=cv2.INTER_CUBIC))

for imagen_valid in valid_tabular_data['img_path']:
    valid_image_data.append(cv2.resize(cv2.imread(imagen_valid,cv2.IMREAD_COLOR),(ancho,alto),
                                          interpolation=cv2.INTER_CUBIC))

for imagen_test in test_tabular_data['img_path']:
    test_image_data.append(cv2.resize(cv2.imread(imagen_test,cv2.IMREAD_COLOR),(ancho,alto),
                                          interpolation=cv2.INTER_CUBIC))

""" Se convierten las imágenes a un array de numpy para poderlas introducir posteriormente en el modelo de red. Además,
se divide todo el array de imágenes entre 255 para escalar los píxeles en el intervalo (0-1). Como resultado, habrá un 
array con forma (X, alto, ancho, canales). """
valid_image_data = (np.array(valid_image_data) / 255)
test_image_data = (np.array(test_image_data) / 255)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN PROCESAMIENTO DE DATOS -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Antes de nada se van a recopilar solo las imagenes de las que se tiene mutado el gen, para utilizarlas después,
cuando se sobremuestree la clase minoritaria con la tecnica SMOTE. """
df_mutations = train_tabular_data.loc[train_tabular_data['CNV'] == 1]
mutation_image_data = []

for image_mutation in df_mutations['img_path']:
    mutation_image_data.append(cv2.resize(cv2.imread(image_mutation, cv2.IMREAD_COLOR), (ancho, alto),
                                           interpolation=cv2.INTER_CUBIC))

""" Una vez ya se tienen las imágenes convertidas en arrays de numpy, se puede eliminar de los dos subconjuntos tanto la
columna 'ID' como la columna 'path_img' que no son útiles para la red MLP. En el caso del subconjunto de entrenamiento,
se guardan ambas columnas para usarlas posteriormente como referencia: """
#@inplace = True para que devuelva el resultado en la misma variable
train_tabular_data.drop(['ID'], axis=1, inplace= True)
train_tabular_data.drop(['img_path'], axis=1, inplace= True)

valid_tabular_data.drop(['ID'], axis=1, inplace= True)
valid_tabular_data.drop(['img_path'], axis=1, inplace= True)

test_tabular_data.drop(['ID'], axis=1, inplace= True)
test_tabular_data.drop(['img_path'], axis=1, inplace= True)

""" Se extrae la columna 'CNV' del dataframe de ambos subconjuntos, puesto que ésta es la salida del modelo que se va a 
entrenar."""
train_labels = train_tabular_data.pop('CNV')
valid_labels = valid_tabular_data.pop('CNV')
test_labels = test_tabular_data.pop('CNV')

""" Ahora se procede a procesar las columnas continuas, que se escalarán para que estén en el rango de (0-1), es decir, 
como la salida de la red. """
scaler = MinMaxScaler()

""" Hay 'warning' si se hace directamente, así que se hace de esta manera. Se transforman los datos guardándolos en una
variable. Posteriormente se modifica la columna de las tablas con esa variable. """
train_continuous = scaler.fit_transform(train_tabular_data[['Age']])
valid_continuous = scaler.transform(valid_tabular_data[['Age']])
test_continuous = scaler.transform(test_tabular_data[['Age']])

train_tabular_data.loc[:,'Age'] = train_continuous[:,0]
valid_tabular_data.loc[:,'Age'] = valid_continuous[:,0]
test_tabular_data.loc[:,'Age'] = test_continuous[:,0]

""" Oversampling. Para intentar corregir el desbalance de clases se va a utilizar la tecnica SMOTE para generar muestras
de la clase minoritaria. Habra el mismo numero de muestras para las dos clases """
smote = SMOTE(sampling_strategy= 'minority')
train_tabular_data_smote, train_labels_smote = smote.fit_resample(X = train_tabular_data, y = train_labels)
oversampling_number = len(train_tabular_data_smote) - len(train_tabular_data)

""" Ahora hay que igualar el numero de imagenes para que se corresponda con el numero de muestras. Todas las muestras
añadidas son de la clase minoritaria y se añaden al final de la ultima fila, por lo que solo habra que añadir imagenes 
de esta clase al final del array de imagenes hasta igualar el numero de muestras."""
difference = oversampling_number - len(mutation_image_data)

for image in mutation_image_data:
    rotate = iaa.Affine(rotate=(-20, 20), mode= 'edge')
    mutation_image_data.append(rotate.augment_image(image))
    difference-= 1
    if difference <= 0:
        break
    gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
    mutation_image_data.append(gaussian_noise.augment_image(image))
    difference-= 1
    if difference <= 0:
        break
    crop = iaa.Crop(percent=(0, 0.3))
    mutation_image_data.append(crop.augment_image(image))
    difference-= 1
    if difference <= 0:
        break
    shear = iaa.Affine(shear=(0, 40), mode= 'edge')
    mutation_image_data.append(shear.augment_image(image))
    difference-= 1
    if difference <= 0:
        break
    flip_hr = iaa.Fliplr(p=1.0)
    mutation_image_data.append(flip_hr.augment_image(image))
    difference-= 1
    if difference <= 0:
        break
    flip_vr = iaa.Flipud(p=1.0)
    mutation_image_data.append(flip_vr.augment_image(image))
    difference-= 1
    if difference <= 0:
        break
    contrast = iaa.GammaContrast(gamma=2.0)
    mutation_image_data.append(contrast.augment_image(image))
    difference-= 1
    if difference <= 0:
        break
    scale_im = iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
    mutation_image_data.append(scale_im.augment_image(image))
    difference-= 1
    if difference <= 0:
        break

""" Una vez hecho esto, se unen las listas de las imágenes que ya teniamos con las imagenes exclusivamente que
tienen mutacion en el gen: """
pre_train_image_data = pre_train_image_data + mutation_image_data

""" Se hace data augmentation a todas las imágenes:"""
train_image_data = []

for image in pre_train_image_data:
    train_image_data.append(image)
    #rotate = iaa.Affine(rotate=(-20, 20), mode= 'edge')
    #train_image_data.append(rotate.augment_image(image))
    gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
    train_image_data.append(gaussian_noise.augment_image(image))
    crop = iaa.Crop(percent=(0, 0.3))
    train_image_data.append(crop.augment_image(image))
    #shear = iaa.Affine(shear=(0, 40), mode= 'edge')
    #train_image_data.append(shear.augment_image(image))
    flip_hr = iaa.Fliplr(p=1.0)
    train_image_data.append(flip_hr.augment_image(image))
    #flip_vr = iaa.Flipud(p=1.0)
    #train_image_data.append(flip_vr.augment_image(image))
    contrast = iaa.GammaContrast(gamma=2.0)
    train_image_data.append(contrast.augment_image(image))
    #scale_im = iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
    #train_image_data.append(scale_im.augment_image(image))

train_image_data = (np.array(train_image_data) / 255)

""" Una vez se tienen hechos los recortes de imágenes, se procede a replicar las filas de ambos subconjuntos de datos
para que el número de imágenes utilizadas y el número de filas del marco de datos sea el mismo: """
# @squeeze = Para transformar una columna de un dataframe en una serie de pandas
# @rename = Para cambiarle el nombre a una serie de pandas
train_tabular_data_smote = pd.DataFrame(np.repeat(train_tabular_data_smote.values, 5, axis=0),
                                        columns=train_tabular_data_smote.columns)
train_labels_smote = pd.DataFrame(np.repeat(train_labels_smote.values, 5, axis=0))
train_labels_smote = train_labels_smote.squeeze().rename('CNV')

""" Para poder entrenar la red hace falta transformar los dataframes de entrenamiento y test en arrays de numpy, así 
como también la columna de salida de ambos subconjuntos (las imágenes YA fueron convertidas anteriormente, por lo que no
hace falta transformarlas de nuevo). """
train_tabular_data_smote = np.asarray(train_tabular_data_smote).astype('float32')
train_labels_smote = np.asarray(train_labels_smote).astype('float32')

valid_tabular_data = np.asarray(valid_tabular_data).astype('float32')
valid_labels = np.asarray(valid_labels).astype('float32')

test_tabular_data = np.asarray(test_tabular_data).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (MLP + CNN) -----------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Se definen dos entradas para dos ramas distintas: una para la red perceptrón multicapa; y otra para la red neuronal 
convolucional """
inputA = Input(shape = (train_tabular_data.shape[1]),)
inputB = Input(shape = (alto, ancho, canales))

""" La primera rama (Perceptrón multicapa) opera con la primera entrada: """
x = layers.Dense(train_tabular_data.shape[1], activation = "relu")(inputA)
x = layers.Dropout(0.3)(x)
x = layers.Dense(28, activation = "relu")(x)
x = keras.models.Model(inputs = inputA, outputs = x)

""" La segunda rama (Red neuronal convolucional) opera con la segunda entrada: """
# @training = False: Para que el modelo se ejecute en modo inferencia. Es importante para el posterior 'fine-tuning'
# @Dropout: Capa de regularización para no favorecer el sobreentrenamiento
# @GlobalAveragePooling2D: Convierte características de 'inputB.output_shape[1:]' en vectores
y = layers.SeparableConv2D(32, (3, 3), padding="same", activation='relu')(inputB)
y = layers.BatchNormalization(axis=3)(y)
y = layers.MaxPooling2D(pool_size=(2, 2))(y)
y = layers.Dropout(0.25)(y)

y = layers.SeparableConv2D(64, (3, 3), padding="same", activation='relu')(y)
y = layers.BatchNormalization(axis=3)(y)
y = layers.SeparableConv2D(64, (3, 3), padding="same", activation='relu')(y)
y = layers.BatchNormalization(axis=3)(y)
y = layers.MaxPooling2D(pool_size=(2, 2))(y)
y = layers.Dropout(0.25)(y)

y = layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu')(y)
y = layers.BatchNormalization(axis=3)(y)
y = layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu')(y)
y = layers.BatchNormalization(axis=3)(y)
y = layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu')(y)
y = layers.BatchNormalization(axis=3)(y)
y = layers.MaxPooling2D(pool_size=(2, 2))(y)
y = layers.Dropout(0.25)(y)

y = layers.Flatten()(y)
y = layers.Dense(256, activation= 'relu') (y)
y = layers.BatchNormalization()(y)
y = layers.Dropout(0.5)(y)
y = keras.models.Model(inputs = inputB, outputs = y)

""" Se crea la entrada al conjunto final de capas, que será la concatenación de la salida de ambas ramas (la de la red 
perceptrón multicapa y la de la red neuronal convolucional). """
combined = keras.layers.concatenate([x.output, y.output])

""" Una vez se ha concatenados la salida de ambas ramas, se aplica dos capas densamente conectadas, la última de ellas
siendo la de la predicción final con activación sigmoid, puesto que la salida será binaria (0 o 1) """
z = layers.Dense(64, activation="relu")(combined)
z = layers.Dropout(0.5)(z)
z = layers.Dense(1, activation="sigmoid")(z)

""" El modelo final aceptará datos numéricos/categóricos en la entrada de la red perceptrón multicapa e imágenes en la
red neuronal convolucional, de forma que a la salida solo se obtenga un solo valor, la predicción de la mutación del
gen. """
model = keras.models.Model(inputs=[x.input, y.input], outputs=z)
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
checkpoint_path = 'model_snv_MTOR_epoch{epoch:02d}.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = False)

""" Esto se hace para que al hacer el entrenamiento, los pesos de las distintas salidas se balanceen, ya que el conjunto
de datos que se tratan en este problema es muy imbalanceado. """
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels_smote),
                                                  y = train_labels_smote)
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
neural_network = model.fit(x = [train_tabular_data_smote, train_image_data],  # Datos de entrada.
                           y = train_labels_smote,  # Datos objetivos.
                           epochs = 4,
                           verbose = 1,
                           batch_size= 32,
                           class_weight= class_weight_dict,
                           #callbacks= mcp_save,
                           validation_data = ([valid_tabular_data, valid_image_data], valid_labels)) # Datos de validación.

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
#model = keras.models.load_model('model_cnv_pik3ca_epoch{epoch:02d}-recall{val_recall:.2f}-precision{val_precision:.2f}.h5')
results = model.evaluate([test_tabular_data, test_image_data],test_labels, verbose = 0)
print("\n'Loss' del conjunto de prueba: {:.2f}\n""Sensibilidad del conjunto de prueba: {:.2f}\n" 
      "Precisión del conjunto de prueba: {:.2f}\n""Exactitud del conjunto de prueba: {:.2f} %\n"
      "El AUC ROC del conjunto de prueba es de: {:.2f}".format(results[0],results[5],results[6],results[7] * 100,
                                                               results[8]))

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
print("Predicciones:\n", np.round(model.predict([test_tabular_data[:10], test_image_data[:10]])))

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
y_true = test_labels # Etiquetas verdaderas de 'test'
y_pred = np.round(model.predict([test_tabular_data, test_image_data])) # Predicción de etiquetas de 'test'

matrix = confusion_matrix(y_true, y_pred) # Calcula (pero no dibuja) la matriz de confusión

group_names = ['True Neg','False Pos','False Neg','True Pos'] # Nombres de los grupos
group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()] # Cantidad de casos por grupo

""" @zip: Une las tuplas del nombre de los grupos con la de la cantidad de casos por grupo """
labels = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
plt.show() # Muestra la gráfica de la matriz de confusión

#np.save('test_data', test_tabular_data)
#np.save('test_image', test_image_data)
#np.save('test_labels', test_labels)