import shelve # datos persistentes
import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
import staintools
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

""" Se crea un dataframe para el diccionario de mutaciones SNV y otro para el diccionario de la categoría N del sistema
de estadificación TNM. Posteriormente, se renombran las dos columnas para que todo quede más claro y se fusionan ambos
dataframes. En una columna tendremos el ID del paciente, en otra las distintas mutaciones SNV y en la otra la 
categoría N para dicho paciente. """
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)

df_list = [df_path_n_stage]

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
                            (df_all_merge["path_n_stage"]!='N0 (MOL+)')]

""" Directorios de imágenes con cáncer y sin cáncer: """
image_dir = '/home/avalderas/Descargas/Imgspatients'

""" Se seleccionan todas las rutas de las imágenes que tienen cáncer: """
cancer_dir = glob.glob(image_dir + "/*") # 1702 imágenes con cáncer en total

""" Se crea una serie sobre el directorio de las imágenes con cáncer que crea un array de 1-D (columna) en el que en 
cada fila hay una ruta para cada una de las imágenes con cáncer. Posteriormente, se extrae el 'ID' de cada ruta de cada
imagen y se establecen como índices de la serie, por lo que para cada ruta de la serie, ésta tendrá como índice de fila
su 'ID' correspondiente ({TCGA-A7-A13E  C:\...\...\TCGA-A7-A13E-01Z-00-DX2.3.JPG}). 
Por último, el dataframe se une con la serie creada mediante la columna 'ID'. De esta forma, cada paciente replicará sus 
filas el número de veces que tenga una imagen distinta, es decir, que si un paciente tiene 3 imágenes, la fila de datos
de ese paciente se presenta 3 veces, teniendo en cada una de ellas una ruta de imagen distinta: """
series_img = pd.Series(cancer_dir)
series_img.index = series_img.str.extract(fr"({'|'.join(df_all_merge['ID'])})", expand=False)

df_all_merge = df_all_merge.join(series_img.rename('img_path'), on='ID')
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Una vez se tienen todas las imágenes y quitados los valores nulos, tambiés es necesario de aquellas imágenes que son
intraoperatorias. Para ello nos basamos en el archivo 'Pacientes_MGR' para eliminar algunas filas de aquellas imágenes
de algunos pacientes que no nos van a servir. """
# 1672 filas resultantes (al haber menos columnas hay menos valores nulos):
remove_img_list = ['TCGA-A2-A0EW', 'TCGA-E2-A153', 'TCGA-E2-A15A', 'TCGA-E2-A15E', 'TCGA-E9-A1N4', 'TCGA-E9-A1N5',
                   'TCGA-E9-A1N6', 'TCGA-E9-A1NC', 'TCGA-E9-A1ND', 'TCGA-E9-A1NE', 'TCGA-E9-A1NH', 'TCGA-PL-A8LX',
                   'TCGA-PL-A8LZ']

for id_img in remove_img_list:
    index_df_all_merge = df_all_merge.loc[df_all_merge['ID'] == id_img].index
    df_all_merge.drop(index_df_all_merge, inplace=True)

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
target = staintools.read_image('/images/img_lote1_cancer/TCGA-A2-A25D-01Z-00-DX1.2.JPG')
target = staintools.LuminosityStandardizer.standardize(target)
target = normalizer.fit(target)

""" Se dividen las imágenes en teselas del mismo tamaño (en este caso ancho = 7 x 120 y alto = 3 x 210), por lo que al 
final se tienen un total de 21 teselas por imagen. Por otra parte, también se multiplican las etiquetas de salida 
dependiendo del número de teselas en el que se ha dividido la imagen. De esta forma, cada imagen tendrá N teselas, y 
también N filas en las etiquetas de salida, para que así cada tesela esté etiquetada correctamente dependiendo de la 
imagen de la que provenía. """
merge_image_tile = []
merge_image_data = []
merge_labels_tile = []
merge_labels = []

for index_normal_merge, image_merge in enumerate(df_all_merge['img_path']):
    series_img = pd.Series(image_merge)
    series_img.index = series_img.str.extract(fr"({'|'.join(df_all_merge['ID'])})", expand=False) # Nombre ID

    #merge_image_resize = staintools.read_image(image_merge)
    #merge_image_resize = staintools.LuminosityStandardizer.standardize(merge_image_resize)
    #merge_image_resize = normalizer.transform(merge_image_resize)
    #merge_image_resize = cv2.resize(merge_image_resize, (ancho, alto), interpolation = cv2.INTER_CUBIC)
    """ En caso de no querer normalizar las imagenes, se abren directamente y se redimensionan con OpenCV y se eliminan
    las cuatro lineas anteriores"""
    merge_image_resize = cv2.resize(cv2.imread(image_merge,cv2.IMREAD_COLOR),(ancho, alto),interpolation=cv2.INTER_CUBIC)
    merge_tiles = [merge_image_resize[x:x + 210, y:y + 210] for x in range(0, merge_image_resize.shape[0], 210) for y in
                   range(0, merge_image_resize.shape[1], 210)]

    for index_tile, merge_tile in enumerate(merge_tiles):
        print('tile_{}_img{}_n{}'.format(str(series_img.index.values[0]),index_normal_merge, index_tile))
        cv2.imwrite('/home/avalderas/img_slides/tiles_patients_unnormalized/tile_{}_img{}_n{}.JPG'.format(str(series_img.index.values[0]),index_normal_merge, index_tile), merge_tile)