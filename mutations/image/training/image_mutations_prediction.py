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
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_snv = pd.DataFrame.from_dict(snv.items()); df_snv.rename(columns = {0 : 'ID', 1 : 'SNV'}, inplace = True)
df_cnv = pd.DataFrame.from_dict(cnv.items()); df_cnv.rename(columns = {0 : 'ID', 1 : 'CNV'}, inplace = True)

df_list = [df_path_n_stage, df_snv, df_cnv]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

""" Ahora se va a encontrar cuales son los ID de los genes que nos interesa. Para empezar se carga el archivo excel 
donde aparecen todos los genes con mutaciones que interesan estudiar usando 'openpyxl' y creamos una lista para
los genes CNV."""
mutations_target = pd.read_excel('/home/avalderas/img_slides/excel_genesOCA&inibica_patients/Panel_OCA.xlsx',
                                 usecols= 'B:C', engine= 'openpyxl')

snv = mutations_target.loc[mutations_target['Scope'] != 'CNV', 'Gen']
cnv = mutations_target.loc[mutations_target['Scope'] == 'CNV', 'Gen']

# SNV
snv_list = []
for gen_snv in snv:
    if gen_snv not in snv_list:
        snv_list.append(gen_snv)

# CNV
cnv_list = []
for gen_cnv in cnv:
    if gen_cnv not in cnv_list:
        cnv_list.append(gen_cnv)

""" Ahora se recopilan en una lista los distintos IDs de los genes a estudiar. """
id_snv_list = []
id_cnv_list = []

key_list = list(dict_genes.keys())
val_list = list(dict_genes.values())

# SNV
for gen_snv in snv_list:
    position = val_list.index(gen_snv) # Número
    id_gen_snv = (key_list[position]) # Número
    id_snv_list.append(id_gen_snv) # Se añaden todos los IDs en la lista vacía

#CNV
for gen_cnv in cnv_list:
    if gen_cnv == 'RICTOR':
        id_cnv_list.append(253260)
        continue
    position = val_list.index(gen_cnv) # Número
    id_gen_cnv = (key_list[position]) # Número
    id_cnv_list.append(id_gen_cnv) # Se añaden todos los IDs en la lista vacía

""" Ahora se hace un bucle sobre la columna de mutaciones CNV del dataframe. Así, se busca en cada mutación de 
cada fila para ver en cuales de estas filas se encuentra el ID de los genes a estudiar. Se almacenan en una lista de 
listas los índices de las filas donde se encuentran los IDs de esos genes, de forma que se tiene una lista para cada gen. """
""" Se crea esta lista de los pacientes que tienen mutación 'CNV' de estos genes porque hay un fallo en el diccionario 
de mutaciones 'CNV' y no identifica sus mutaciones. Por tanto, se ha recopilado manualmente los 'IDs' de los pacientes
que tienen mutaciones en el gen (gracias a cBioPortal) para poner un '1' en la columna 'CNV' de esos 'IDs'. """
cdkn1b_list_amp = ['TCGA-A1-A0SK', 'TCGA-A1-A0SP', 'TCGA-A2-A04T', 'TCGA-A2-A04U', 'TCGA-A7-A4SD', 'TCGA-A7-A6VW',
                   'TCGA-AN-A0FJ', 'TCGA-AQ-A54N', 'TCGA-C8-A12L', 'TCGA-C8-A1HJ', 'TCGA-E9-A22G']
cdkn1b_list_del = ['TCGA-A2-A3XT', 'TCGA-A8-A06R', 'TCGA-AC-A2FM', 'TCGA-AN-A0AJ', 'TCGA-AR-A24M', 'TCGA-LL-A8F5',
                   'TCGA-OL-A5RU']

brca2_list_amp = ['TCGA-A2-A04T', 'TCGA-A8-A06R', 'TCGA-AN-A0AS', 'TCGA-BH-A0GZ', 'TCGA-BH-A1EV', 'TCGA-D8-A1Y2',
                  'TCGA-E2-A14T', 'TCGA-E2-A1LG']
brca2_list_del = ['TCGA-A7-A0CE', 'TCGA-A8-A08I', 'TCGA-A8-A09V', 'TCGA-A8-A0AB', 'TCGA-AN-A04D', 'TCGA-AR-A24H',
                  'TCGA-B6-A0IQ', 'TCGA-D8-A147', 'TCGA-D8-A1JB', 'TCGA-D8-A1JD', 'TCGA-EW-A1OX', 'TCGA-EW-A1P7',
                  'TCGA-PE-A5DC', 'TCGA-S3-AA10']

brca1_list_amp = ['TCGA-A2-A0EO', 'TCGA-A7-A13D', 'TCGA-A8-A09G', 'TCGA-AC-A2FB', 'TCGA-AN-A04C', 'TCGA-AR-A24H',
                  'TCGA-B6-A0IG', 'TCGA-B6-A0IN', 'TCGA-BH-A42T', 'TCGA-C8-A9FZ', 'TCGA-E2-A105', 'TCGA-E9-A1RI',
                  'TCGA-LD-A9QF']
brca1_list_del = ['TCGA-BH-A0AW', 'TCGA-BH-A0C0', 'TCGA-C8-A12L', 'TCGA-E2-A1L7', 'TCGA-EW-A1OX']

kdr_list_amp = ['TCGA-A2-A04T', 'TCGA-A2-A0YE', 'TCGA-B6-A0RS', 'TCGA-EW-A1P8']
kdr_list_del = ['TCGA-AC-A5EH']

chek1_list_amp = ['TCGA-AR-A2LJ']
chek1_list_del = ['TCGA-A8-A0A1', 'TCGA-BH-A18M', 'TCGA-BH-A1FN', 'TCGA-C8-A130', 'TCGA-D8-A147', 'TCGA-E2-A56Z',
                  'TCGA-E2-A9RU', 'TCGA-E9-A1RF', 'TCGA-EW-A1OX', 'TCGA-LL-A6FP']

fanca_list_amp = ['TCGA-A2-A04P', 'TCGA-A2-A0D2', 'TCGA-AO-A0J2', 'TCGA-EW-A1PB']
fanca_list_del = ['TCGA-A1-A0SG', 'TCGA-A2-A0D1', 'TCGA-A7-A0CD', 'TCGA-A7-A0CH', 'TCGA-A7-A5ZW', 'TCGA-A8-A08H',
                  'TCGA-A8-A09V', 'TCGA-A8-A0A1', 'TCGA-AC-A3YI', 'TCGA-AC-A62V', 'TCGA-AO-A0JC', 'TCGA-AR-A2LQ',
                  'TCGA-B6-A0IM', 'TCGA-B6-A0RM', 'TCGA-BH-A0AU', 'TCGA-BH-A0BF', 'TCGA-BH-A18J', 'TCGA-BH-A18M',
                  'TCGA-BH-A18U', 'TCGA-BH-A1FB', 'TCGA-BH-A28O', 'TCGA-C8-A12T', 'TCGA-D8-A73X', 'TCGA-E2-A15J',
                  'TCGA-E9-A295', 'TCGA-EW-A1IY', 'TCGA-EW-A1PG', 'TCGA-GM-A5PV', 'TCGA-OL-A6VO', 'TCGA-S3-AA10']

rictor_list_amp = ['TCGA-A2-A0D0', 'TCGA-A2-A25B', 'TCGA-A7-A13D', 'TCGA-A8-A09C', 'TCGA-AC-A2FE', 'TCGA-AR-A0TW',
                  'TCGA-B6-A3ZX', 'TCGA-BH-A0BP', 'TCGA-BH-A1FU', 'TCGA-C8-A131', 'TCGA-D8-A27H', 'TCGA-E2-A574']
rictor_list_del = ['TCGA-BH-A0B3', 'TCGA-GM-A3XL']

""" Se recopila los índices de las distintas filas donde aparecen las mutaciones 'CNV' de los genes seleccionados (tanto 
de amplificación como deleción), y se añaden a la lista de listas correspondiente (la de amplificación o la de deleción). """
# SNV
list_gen_snv = [[] for ID in range(len(id_snv_list))]

for index, id_snv in enumerate (id_snv_list): # Para cada ID del gen SNV de la lista...
    for index_row, row in enumerate (df_all_merge['SNV']): # Para cada fila dentro de la columna 'SNV'...
        for mutation in row: # Para cada mutación dentro de cada fila...
            if mutation[1] == id_snv: # Si el ID de la mutación es el mismo que el ID de la lista de genes...
                list_gen_snv[index].append(index_row) # Se almacena el índice de la fila en la lista de listas
                
# CNV
list_gen_cnv_amp = [[] for ID in range(len(id_cnv_list))]
list_gen_cnv_del = [[] for ID in range(len(id_cnv_list))]

for index, id_cnv in enumerate (id_cnv_list): # Para cada ID del gen CNV de la lista...
    if id_cnv == 1027: # CDKN1B
        for patient_cdkn1b_amp in cdkn1b_list_amp:
            for index_cdkn1b_amp, row_cdkn1b_amp in enumerate(df_all_merge['ID']):
                if patient_cdkn1b_amp == row_cdkn1b_amp:
                    list_gen_cnv_amp[index].append(index_cdkn1b_amp)
        for patient_cdkn1b_del in cdkn1b_list_del:
            for index_cdkn1b_del, row_cdkn1b_del in enumerate(df_all_merge['ID']):
                if patient_cdkn1b_del == row_cdkn1b_del:
                    list_gen_cnv_del[index].append(index_cdkn1b_del)

    elif id_cnv == 675: # BRCA2
        for patient_brca2_amp in brca2_list_amp:
            for index_brca2_amp, row_brca2_amp in enumerate(df_all_merge['ID']):
                if patient_brca2_amp == row_brca2_amp:
                    list_gen_cnv_amp[index].append(index_brca2_amp)
        for patient_brca2_del in brca2_list_del:
            for index_brca2_del, row_brca2_del in enumerate(df_all_merge['ID']):
                if patient_brca2_del == row_brca2_del:
                    list_gen_cnv_del[index].append(index_brca2_del)

    elif id_cnv == 672: # BRCA1
        for patient_brca1_amp in brca1_list_amp:
            for index_brca1_amp, row_brca1_amp in enumerate(df_all_merge['ID']):
                if patient_brca1_amp == row_brca1_amp:
                    list_gen_cnv_amp[index].append(index_brca1_amp)
        for patient_brca1_del in brca1_list_del:
            for index_brca1_del, row_brca1_del in enumerate(df_all_merge['ID']):
                if patient_brca1_del == row_brca1_del:
                    list_gen_cnv_del[index].append(index_brca1_del)

    elif id_cnv == 3791: # KDR
        for patient_kdr_amp in kdr_list_amp:
            for index_kdr_amp, row_kdr_amp in enumerate(df_all_merge['ID']):
                if patient_kdr_amp == row_kdr_amp:
                    list_gen_cnv_amp[index].append(index_kdr_amp)
        for patient_kdr_del in kdr_list_del:
            for index_kdr_del, row_kdr_del in enumerate(df_all_merge['ID']):
                if patient_kdr_del == row_kdr_del:
                    list_gen_cnv_del[index].append(index_kdr_del)

    elif id_cnv == 1111: # CHEK1
        for patient_chek1_amp in chek1_list_amp:
            for index_chek1_amp, row_chek1_amp in enumerate(df_all_merge['ID']):
                if patient_chek1_amp == row_chek1_amp:
                    list_gen_cnv_amp[index].append(index_chek1_amp)
        for patient_chek1_del in chek1_list_del:
            for index_chek1_del, row_chek1_del in enumerate(df_all_merge['ID']):
                if patient_chek1_del == row_chek1_del:
                    list_gen_cnv_del[index].append(index_chek1_del)

    elif id_cnv == 2175: # FANCA
        for patient_fanca_amp in fanca_list_amp:
            for index_fanca_amp, row_fanca_amp in enumerate(df_all_merge['ID']):
                if patient_fanca_amp == row_fanca_amp:
                    list_gen_cnv_amp[index].append(index_fanca_amp)
        for patient_fanca_del in fanca_list_del:
            for index_fanca_del, row_fanca_del in enumerate(df_all_merge['ID']):
                if patient_fanca_del == row_fanca_del:
                    list_gen_cnv_del[index].append(index_fanca_del)

    elif id_cnv == 253260: # RICTOR
        for patient_rictor_amp in rictor_list_amp:
            for index_rictor_amp, row_rictor_amp in enumerate(df_all_merge['ID']):
                if patient_rictor_amp == row_rictor_amp:
                    list_gen_cnv_amp[index].append(index_rictor_amp)
        for patient_rictor_del in rictor_list_del:
            for index_rictor_del, row_rictor_del in enumerate(df_all_merge['ID']):
                if patient_rictor_del == row_rictor_del:
                    list_gen_cnv_del[index].append(index_rictor_del)

    else:
        for index_row, row in enumerate (df_all_merge['CNV']): # Para cada fila dentro de la columna 'CNV'...
            for mutation in row: # Para cada mutación dentro de cada fila...
                if mutation[1] == id_cnv and mutation[2] > 0:
                    list_gen_cnv_amp[index].append(index_row) # Se almacena el índice de la fila en la lista de listas
                elif mutation[1] == id_cnv and mutation[2] < 0:
                    list_gen_cnv_del[index].append(index_row) # Se almacena el índice de la fila en la lista de listas

""" Una vez se tienen almacenados los índices de las filas donde se producen las mutaciones CNV, hay que crear distintas 
columnas para cada uno de los genes objetivo, para asi mostrar la informacion de uno en uno. De esta forma, habra dos 
columnas distintas para cada gen CNV a estudiar (amplificacion y delecion). Ademas, se recopilan las columnas creadas en 
listas (dos para las columnas de mutaciones CNV). """
columns_list_snv = []

columns_list_cnv_amp = []
columns_list_cnv_normal = []
columns_list_cnv_del = []

# SNV:
df_all_merge.drop(['SNV'], axis=1, inplace= True)
for gen_snv in snv_list:
    df_all_merge['SNV_' + gen_snv] = 0
    columns_list_snv.append('SNV_' + gen_snv)

# CNV 
df_all_merge.drop(['CNV'], axis=1, inplace= True)
for gen_cnv in cnv_list:
    df_all_merge['CNV_' + gen_cnv + '_AMP'] = 0
    df_all_merge['CNV_' + gen_cnv + '_NORMAL'] = 0
    df_all_merge['CNV_' + gen_cnv + '_DEL'] = 0
    columns_list_cnv_amp.append('CNV_' + gen_cnv + '_AMP')
    columns_list_cnv_normal.append('CNV_' + gen_cnv + '_NORMAL')
    columns_list_cnv_del.append('CNV_' + gen_cnv + '_DEL')

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

# CNV
i_cnv_amp = 0
for column_cnv_amp in columns_list_cnv_amp:
    for index_cnv_sublist_amp in list_gen_cnv_amp[i_cnv_amp]:
        df_all_merge.loc[index_cnv_sublist_amp, column_cnv_amp] = 1
    i_cnv_amp += 1

i_cnv_del = 0
for column_cnv_del in columns_list_cnv_del:
    for index_cnv_sublist_del in list_gen_cnv_del[i_cnv_del]:
        df_all_merge.loc[index_cnv_sublist_del, column_cnv_del] = 1
    i_cnv_del += 1

""" Falta por rellenar la columna normal de todas las mutaciones CNV. Para ello se colocará un '1' en aquellas filas 
donde no haya mutación CNV ni de amplificación ni de deleción para un gen en especifico. """
i_cnv_normal = 0
for column_cnv_normal in columns_list_cnv_normal:
    for i_row in range(1084):
        if i_row not in list_gen_cnv_amp[i_cnv_normal] and i_row not in list_gen_cnv_del[i_cnv_normal]:
            df_all_merge.loc[i_row, column_cnv_normal] = 1
    i_cnv_normal += 1

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
train_data, valid_data = train_test_split(train_data, test_size = 0.15)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------------- SECCIÓN IMÁGENES -------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Directorios de teselas con cáncer normalizadas: """
image_dir = '/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer'

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
alto = int(210) # Eje Y: 630. Nº de filas. 210 x 3 = 630
ancho = int(210) # Eje X: 1480. Nº de columnas. 210 x 7 = 1470
canales = 3 # Imágenes a color (RGB) = 3

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

""" Se extraen las etiquetas de salida para cada tipo de mutación """
train_labels_snv = train_data.iloc[:, 2:153]
valid_labels_snv = valid_data.iloc[:, 2:153]
test_labels_snv = test_data.iloc[:, 2:153]

train_labels_cnv_a = train_data.iloc[:, 153::3]
valid_labels_cnv_a = valid_data.iloc[:, 153::3]
test_labels_cnv_a = test_data.iloc[:, 153::3]

train_labels_cnv_d = train_data.iloc[:, 155::3]
valid_labels_cnv_d = valid_data.iloc[:, 155::3]
test_labels_cnv_d = test_data.iloc[:, 155::3]

train_labels_cnv_normal = train_data.iloc[:, 154::3]
valid_labels_cnv_normal = valid_data.iloc[:, 154::3]
test_labels_cnv_normal = test_data.iloc[:, 154::3]

""" Se extraen los nombres de las columnas de las clases de salida para usarlos posteriormente en la sección de
evaluación: """
test_columns_snv = test_labels_snv.columns.values
classes_snv = test_columns_snv.tolist()

test_columns_cnv_a = test_labels_cnv_a.columns.values
classes_cnv_a = test_columns_cnv_a.tolist()

test_columns_cnv_d = test_labels_cnv_d.columns.values
classes_cnv_d = test_columns_cnv_d.tolist()

test_columns_cnv_normal = test_labels_cnv_normal.columns.values
classes_cnv_normal = test_columns_cnv_normal.tolist()

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada, y se recopila la longitud de las imágenes de
entrenamiento y validacion para utilizarlas posteriormente en el entrenamiento: """
del df_all_merge, df_path_n_stage, df_list

train_image_data_len = len(train_image_data)
valid_image_data_len = len(valid_image_data)
batch_dimension = 32

""" Para poder entrenar la red hace falta transformar las tablas en arrays. Para ello se utiliza 'numpy'. Las imágenes 
YA están convertidas en 'arrays' numpy """
train_labels_snv = np.asarray(train_labels_snv).astype('float32')
train_labels_cnv_a = np.asarray(train_labels_cnv_a).astype('float32')
train_labels_cnv_d = np.asarray(train_labels_cnv_d).astype('float32')
train_labels_cnv_normal = np.asarray(train_labels_cnv_normal).astype('float32')

valid_labels_snv = np.asarray(valid_labels_snv).astype('float32')
valid_labels_cnv_a = np.asarray(valid_labels_cnv_a).astype('float32')
valid_labels_cnv_d = np.asarray(valid_labels_cnv_d).astype('float32')
valid_labels_cnv_normal = np.asarray(valid_labels_cnv_normal).astype('float32')

test_labels_snv = np.asarray(test_labels_snv).astype('float32')
test_labels_cnv_a = np.asarray(test_labels_cnv_a).astype('float32')
test_labels_cnv_d = np.asarray(test_labels_cnv_d).astype('float32')
test_labels_cnv_normal = np.asarray(test_labels_cnv_normal).astype('float32')

""" Se pueden guardar en formato de 'numpy' las imágenes y las etiquetas de test para usarlas después de entrenar la red
neuronal convolucional. """
#np.save('test_image', test_image_data)
#np.save('test_labels_snv', test_labels_snv)
#np.save('test_labels_cnv_a', test_labels_cnv_a)
#np.save('test_labels_cnv_d', test_labels_cnv_d)
#np.save('test_labels_cnv_normal', test_labels_cnv_normal)

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL CONVOLUCIONAL ---------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" En esta ocasión, se crea un modelo secuencial para la red neuronal convolucional que será la encargada de procesar
todas las imágenes: """
base_model = keras.applications.EfficientNetB7(weights = 'imagenet', input_tensor = Input(shape=(alto, ancho, canales)),
                                              include_top = False, pooling = 'max')

all_model = base_model.output
all_model = layers.Flatten()(all_model)
all_model = layers.Dense(512)(all_model)
all_model = layers.Dropout(0.5)(all_model)
all_model = layers.Dense(256)(all_model)
all_model = layers.Dropout(0.5)(all_model)

snv = layers.Dense(train_labels_snv.shape[1], activation = "sigmoid", name= 'snv')(all_model)
cnv_a = layers.Dense(train_labels_cnv_a.shape[1], activation = "sigmoid", name = 'cnv_a')(all_model)
cnv_normal = layers.Dense(train_labels_cnv_normal.shape[1], activation = "sigmoid", name = 'cnv_normal')(all_model)
cnv_d = layers.Dense(train_labels_cnv_d.shape[1], activation = "sigmoid", name= 'cnv_d')(all_model)

model = Model(inputs = base_model.input, outputs = [snv, cnv_a, cnv_normal, cnv_d])

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
           keras.metrics.AUC(curve = 'PR', name='AUC-PR')]

model.compile(loss = {'snv': 'binary_crossentropy', 'cnv_a': 'binary_crossentropy', 'cnv_normal': 'binary_crossentropy',
                      'cnv_d': 'binary_crossentropy'},
              optimizer = keras.optimizers.Adam(learning_rate = 0.00001),
              metrics = metrics)
model.summary()

""" Se implementan varios callbacks para guardar el mejor modelo. """
checkpoint_path = '/home/avalderas/img_slides/mutations/image/inference/models/model_image_mutations_{epoch:02d}_{val_loss:.2f}.h5'
mcp_save = ModelCheckpoint(filepath = checkpoint_path, monitor = 'val_loss', mode = 'min')

""" Una vez definido el modelo, se entrena: """
model.fit(x = train_image_data, y = {'snv': train_labels_snv, 'cnv_a': train_labels_cnv_a,
                                     'cnv_normal': train_labels_cnv_normal, 'cnv_d': train_labels_cnv_d},
          epochs = 2, verbose = 1, validation_data = (valid_image_data, {'snv': valid_labels_snv,
                                                                         'cnv_a': valid_labels_cnv_a,
                                                                         'cnv_normal': valid_labels_cnv_normal,
                                                                         'cnv_d': valid_labels_cnv_d}),
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
model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.00001),
              loss = {'snv': 'binary_crossentropy', 'cnv_a': 'binary_crossentropy', 'cnv_normal': 'binary_crossentropy',
                      'cnv_d': 'binary_crossentropy'},
              metrics = metrics)
model.summary()

""" Una vez descongelado las capas convolucionales seleccionadas y compilado de nuevo el modelo, se entrena otra vez. """
neural_network = model.fit(x = train_image_data, y = {'snv': train_labels_snv, 'cnv_a': train_labels_cnv_a,
                                                      'cnv_normal': train_labels_cnv_normal,
                                                      'cnv_d': train_labels_cnv_d},
                           epochs = 100, verbose = 1, validation_data = (valid_image_data, {'snv': valid_labels_snv,
                                                                                          'cnv_a': valid_labels_cnv_a,
                                                                                          'cnv_normal': valid_labels_cnv_normal,
                                                                                          'cnv_d': valid_labels_cnv_d}),
                           callbacks = mcp_save,
                           steps_per_epoch = (train_image_data_len / batch_dimension),
                           validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data, [test_labels_snv, test_labels_cnv_a, test_labels_cnv_normal,
                                           test_labels_cnv_d], verbose = 0)

print("\n'Loss' de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "SNV del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones SNV del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f} \n"
      "Exactitud de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones SNV"
      " del panel OCA en el conjunto de prueba: {:.2f}\nAUC-PR de las mutaciones SNV del panel OCA en el conjunto de "
      "prueba: {:.2f}".format(results[1], results[9], results[10], results[7]/(results[7]+results[6]),
                              results[11] * 100, results[12], results[13]))

if results[9] > 0 or results[10] > 0:
    print("Valor-F de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[9] * results[10]) /
                                                                                                (results[9] + results[10])))

print("\n'Loss' de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones CNV-A del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n"
      "Exactitud de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones "
      "CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""AUC-PR de las mutaciones CNV-A del panel OCA en el "
      "conjunto de prueba: {:.2f}".format(results[2], results[18], results[19], results[16]/(results[16]+results[15]),
                                          results[20] * 100, results[21], results[22]))

if results[18] > 0 or results[19] > 0:
    print("Valor-F de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[18] * results[19]) /
                                                                                                  (results[18] + results[19])))

print("\n'Loss' de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones CNV-D del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n"
      "Exactitud de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones "
      "CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""AUC-PR de las mutaciones CNV-D del panel OCA en el "
      "conjunto de prueba: {:.2f}".format(results[4], results[36], results[37], results[34]/(results[34]+results[33]),
                                          results[38] * 100, results[39], results[40]))
if results[36] > 0 or results[37] > 0:
    print("Valor-F de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[36] * results[37]) /
                                                                                                  (results[36] + results[37])))

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
""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
# SNV
y_true_snv = test_labels_snv
y_pred_snv = np.round(model.predict(test_image_data)[0])

matrix_snv = multilabel_confusion_matrix(y_true_snv, y_pred_snv)

group_names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
index_snv = 0

for matrix_gen_snv in matrix_snv:
    group_counts = ['{0:0.0f}'.format(value) for value in matrix_gen_snv.flatten()]  # Cantidad de casos por grupo
    true_neg_pos_neg = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
    true_neg_pos_neg = np.asarray(true_neg_pos_neg).reshape(2, 2)
    sns.heatmap(matrix_gen_snv, annot=true_neg_pos_neg, fmt='', cmap='Blues')
    plt.title('Mutación SNV del gen {}'.format(classes_snv[index_snv].split('_')[1]))
    plt.show()
    plt.pause(0.2)
    index_snv = index_snv + 1

# CNV-A
y_true_cnv_a = test_labels_cnv_a
y_pred_cnv_a = np.round(model.predict(test_image_data)[1])

matrix_cnv_a = multilabel_confusion_matrix(y_true_cnv_a, y_pred_cnv_a) # Calcula (pero no dibuja) la matriz de confusión
index_cnv_a = 0

for matrix_gen_cnv_a in matrix_cnv_a:
    group_counts = ['{0:0.0f}'.format(value) for value in matrix_gen_cnv_a.flatten()]  # Cantidad de casos por grupo
    true_neg_pos_neg = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
    true_neg_pos_neg = np.asarray(true_neg_pos_neg).reshape(2, 2)
    sns.heatmap(matrix_gen_cnv_a, annot=true_neg_pos_neg, fmt='', cmap='Blues')
    plt.title('Mutación CNV-A del gen {}'.format(classes_cnv_a[index_cnv_a].split('_')[1]))
    plt.show()
    plt.pause(0.2)
    index_cnv_a = index_cnv_a + 1

# CNV-D
y_true_cnv_d = test_labels_cnv_d
y_pred_cnv_d = np.round(model.predict(test_image_data)[3])

matrix_cnv_d = multilabel_confusion_matrix(y_true_cnv_d, y_pred_cnv_d) # Calcula (pero no dibuja) la matriz de confusión
index_cnv_d = 0

for matrix_gen_cnv_d in matrix_cnv_d:
    group_counts = ['{0:0.0f}'.format(value) for value in matrix_gen_cnv_d.flatten()]  # Cantidad de casos por grupo
    true_neg_pos_neg = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names, group_counts)]
    true_neg_pos_neg = np.asarray(true_neg_pos_neg).reshape(2, 2)
    sns.heatmap(matrix_gen_cnv_d, annot=true_neg_pos_neg, fmt='', cmap='Blues')
    plt.title('Mutación CNV-D del gen {}'.format(classes_cnv_d[index_cnv_d].split('_')[1]))
    plt.show()
    plt.pause(0.2)
    index_cnv_d = index_cnv_d + 1

""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos. """
proba_snv = model.predict(test_image_data[:1])[0]
proba_cnv_a = model.predict(test_image_data[:1])[1]
proba_cnv_d = model.predict(test_image_data[:1])[3]

idxs_snv = np.argsort(proba_snv)[::-1]
idxs_cnv_a = np.argsort(proba_cnv_a)[::-1]
idxs_cnv_d = np.argsort(proba_cnv_d)[::-1]

for (i, j) in enumerate(idxs_snv):
    label_snv = "\nLa mutación SNV más probable es del gen {}: {:.2f}%".format(classes_snv[j].split('_')[1],
                                                                               proba[j] * 100)
    print(label_snv)
    break
    # text = "Most likely SNV mutation: {} gene ({:.2f}%)".format(classes[j].split('_')[1], proba[j] * 100)
    # cv2.putText(test_image_data[:1], text, (870, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 ,0), 2)

for (i, j) in enumerate(idxs_cnv_a):
    label_cnv_a = "\nLa mutación CNV-A más probable es del gen {}: {:.2f}%".format(classes_cnv_a[j].split('_')[1],
                                                                                   proba[j] * 100)
    print(label_cnv_a)
    break
    # text = "Most likely CNV-A mutation: {} gene ({:.2f}%)".format(classes[j].split('_')[1], proba[j] * 100)
    # cv2.putText(test_image_data[:1], text, (870, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 ,0), 2)

for (i, j) in enumerate(idxs_cnv_d):
    label_cnv_d = "\nLa mutación CNV-D más probable es del gen {}: {:.2f}%".format(classes_cnv_d[j].split('_')[1],
                                                                                   proba[j] * 100)
    print(label_cnv_d)
    break
    # text = "Most likely CNV-D mutation: {} gene ({:.2f}%)".format(classes[j].split('_')[1], proba[j] * 100)
    # cv2.putText(test_image_data[:1], text, (870, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 ,0), 2)

""" Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases.
Para implementarlas, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva: """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interp

y_pred_prob_snv = model.predict(test_image_data)[0]
y_pred_prob_cnv_a = model.predict(test_image_data)[1]
y_pred_prob_cnv_d = model.predict(test_image_data)[3]

# SNV
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(classes_snv)):
    if len(np.unique(test_labels_snv[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_snv[:, i], y_pred_prob_snv[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_snv.ravel(), y_pred_prob_snv.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for SNV mutations')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(classes_snv)):
    if len(np.unique(test_labels_snv[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_snv[:, i], y_pred_prob_snv[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_snv.ravel(), y_pred_prob_snv.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve (micro) for SNV mutations')
plt.legend(loc = 'best')
plt.show()

# CNV-A
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(classes_cnv_a)):
    if len(np.unique(test_labels_cnv_a[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_cnv_a[:, i], y_pred_prob_cnv_a[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_cnv_a.ravel(), y_pred_prob_cnv_a.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for CNV-A mutations')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(classes_cnv_a)):
    if len(np.unique(test_labels_cnv_a[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_cnv_a[:, i], y_pred_prob_cnv_a[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_cnv_a.ravel(), y_pred_prob_cnv_a.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for CNV-A mutations')
plt.legend(loc = 'best')
plt.show()

#CNV-D
fpr = dict()
tpr = dict()
auc_roc = dict()

""" Se calcula la tasa de falsos positivos y de verdaderos negativos para cada una de las clases, buscando en cada una
de las 'n' (del número de clases) columnas del problema y se calcula con ello el AUC-ROC micro-promedio """
for i in range(len(classes_cnv_d)):
    if len(np.unique(test_labels_cnv_d[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(test_labels_cnv_d[:, i], y_pred_prob_cnv_d[:, i])
        auc_roc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_cnv_d.ravel(), y_pred_prob_cnv_d.ravel())
auc_roc["micro"] = auc(fpr["micro"], tpr["micro"])

""" Finalmente se dibuja la curva AUC-ROC micro-promedio """
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label = 'Micro-average AUC-ROC curve (AUC = {0:.2f})'.format(auc_roc["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve for CNV-D mutations')
plt.legend(loc = 'best')
plt.show()

""" Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
precision = dict()
recall = dict()
auc_pr = dict()

""" Se calcula precisión y la sensibilidad para cada una de las clases, buscando en cada una de las 'n' (del número de 
clases) columnas del problema y se calcula con ello el AUC-PR micro-promedio """
for i in range(len(classes_cnv_d)):
    if len(np.unique(test_labels_cnv_d[:, i])) > 1:
        precision[i], recall[i], _ = precision_recall_curve(test_labels_cnv_d[:, i], y_pred_prob_cnv_d[:, i])
        auc_pr[i] = auc(recall[i], precision[i])

precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels_cnv_d.ravel(), y_pred_prob_cnv_d.ravel())
auc_pr["micro"] = auc(recall["micro"], precision["micro"])

""" Finalmente se dibuja la curvas AUC-PR micro-promedio """
plt.figure()
plt.plot(recall["micro"], precision["micro"],
         label = 'Micro-average AUC-PR curve (AUC = {0:.2f})'.format(auc_pr["micro"]),
         color = 'blue', linewidth = 2)

plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR curve for CNV-D mutations')
plt.legend(loc = 'best')
plt.show()