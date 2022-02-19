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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split  # Se importa la librería para dividir los datos en entreno y test.
from sklearn.preprocessing import MinMaxScaler  # Para escalar valores
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix  # Para realizar la matriz de confusión

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN DATOS TABULARES ------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
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
    age = data.get('age')
    neoadjuvant = data.get('neoadjuvant') # 1 valor nulo
    prior_diagnosis = data.get('prior_diagnosis') # 1 valor nulo
    # Datos de metástasis a distancia
    os_status = data.get('os_status')
    dfs_status = data.get('dfs_status')  # 142 valores nulos
    tumor_type = data.get('tumor_type')
    stage = data.get('stage')
    path_t_stage = data.get('path_t_stage')
    path_n_stage = data.get('path_n_stage')
    path_m_stage = data.get('path_m_stage')
    subtype = data.get('subtype')
    snv = data.get('mutations')
    cnv = data.get('CNAs')

""" Se crean dataframes individuales para cada uno de los diccionarios almacenados en cada variable de entrada y se
renombran las columnas para que todo quede más claro. Además, se crea una lista con todos los dataframes para
posteriormente unirlos todos juntos. """
df_age = pd.DataFrame.from_dict(age.items()); df_age.rename(columns = {0 : 'ID', 1 : 'Age'}, inplace = True)
df_neoadjuvant = pd.DataFrame.from_dict(neoadjuvant.items()); df_neoadjuvant.rename(columns = {0 : 'ID', 1 : 'Neoadjuvant'}, inplace = True)
df_prior_diagnosis = pd.DataFrame.from_dict(prior_diagnosis.items()); df_prior_diagnosis.rename(columns = {0 : 'ID', 1 : 'prior_diagnosis'}, inplace = True)
# Columna de metástasis a distancia
df_os_status = pd.DataFrame.from_dict(os_status.items()); df_os_status.rename(columns = {0 : 'ID', 1 : 'os_status'}, inplace = True)
df_dfs_status = pd.DataFrame.from_dict(dfs_status.items()); df_dfs_status.rename(columns = {0 : 'ID', 1 : 'dfs_status'}, inplace = True)
df_tumor_type = pd.DataFrame.from_dict(tumor_type.items()); df_tumor_type.rename(columns = {0 : 'ID', 1 : 'tumor_type'}, inplace = True)
df_stage = pd.DataFrame.from_dict(stage.items()); df_stage.rename(columns = {0 : 'ID', 1 : 'stage'}, inplace = True)
df_path_t_stage = pd.DataFrame.from_dict(path_t_stage.items()); df_path_t_stage.rename(columns = {0 : 'ID', 1 : 'path_t_stage'}, inplace = True)
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_path_m_stage = pd.DataFrame.from_dict(path_m_stage.items()); df_path_m_stage.rename(columns = {0 : 'ID', 1 : 'path_m_stage'}, inplace = True)
df_subtype = pd.DataFrame.from_dict(subtype.items()); df_subtype.rename(columns = {0 : 'ID', 1 : 'subtype'}, inplace = True)
df_snv = pd.DataFrame.from_dict(snv.items()); df_snv.rename(columns = {0 : 'ID', 1 : 'SNV'}, inplace = True)
df_cnv = pd.DataFrame.from_dict(cnv.items()); df_cnv.rename(columns = {0 : 'ID', 1 : 'CNV'}, inplace = True)

df_list = [df_age, df_neoadjuvant, df_os_status, df_dfs_status, df_tumor_type, df_stage, df_path_t_stage,
           df_path_n_stage, df_path_m_stage, df_subtype, df_snv, df_cnv]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

""" Se crea una nueva columna para indicar la metastasis a distancia. En esta columna se indicaran los pacientes que 
tienen estadio M1 (metastasis inicial) + otros pacientes que desarrollan metastasis a lo largo de la enfermedad (para
ello se hace uso del excel pacientes_tcga y su columna DB) """
df_all_merge['Distant Metastasis'] = 0
df_all_merge.loc[df_all_merge.path_m_stage == 'M1', 'Distant Metastasis'] = 1

""" Estos pacientes desarrollan metastasis A LO LARGO de la enfermedad, tal y como se puede apreciar en el excel de los
pacientes de TCGA. Por tanto, se incluyen como clase positiva dentro de la columna 'distant_metastasis'. """
df_all_merge.loc[df_all_merge.ID == 'TCGA-A2-A3XS', 'Distant Metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-AC-A2FM', 'Distant Metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-AR-A2LH', 'Distant Metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-BH-A0C1', 'Distant Metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-BH-A18V', 'Distant Metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-EW-A1P8', 'Distant Metastasis'] = 1
df_all_merge.loc[df_all_merge.ID == 'TCGA-GM-A2DA', 'Distant Metastasis'] = 1

""" Se recoloca la columna de metástasis a distancia al lado de la de recaídas para dejar las mutaciones como las 
últimas columnas. """
cols = df_all_merge.columns.tolist()
cols = cols[:5] + cols[-1:] + cols[5:-1]
df_all_merge = df_all_merge[cols]

""" Ahora se va a encontrar cuales son los ID de los genes que nos interesa. Para empezar se carga el archivo excel 
donde aparecen todos los genes con mutaciones que interesan estudiar usando 'openpyxl' y creamos dos listas. Una para
los genes SNV y otra para los genes CNV."""
mutations_target = pd.read_excel('/home/avalderas/img_slides/excels/Panel_OCA.xlsx', usecols= 'B:C', engine= 'openpyxl')

snv = mutations_target.loc[mutations_target['Scope'] != 'CNV', 'Gen']
cnv = mutations_target.loc[mutations_target['Scope'] == 'CNV', 'Gen']

# SNV:
snv_list = []
for gen_snv in snv:
    if gen_snv not in snv_list:
        snv_list.append(gen_snv)

# CNV:
cnv_list = []
for gen_cnv in cnv:
    if gen_cnv not in cnv_list:
        cnv_list.append(gen_cnv)

""" Ahora se recopilan en dos listas (una para SNV y otra para CNV) los distintos IDs de los genes a estudiar. """
id_snv_list = []
id_cnv_list = []

key_list = list(dict_genes.keys())
val_list = list(dict_genes.values())

# SNV:
for gen_snv in snv_list:
    position = val_list.index(gen_snv) # Número
    id_gen_snv = (key_list[position]) # Número
    id_snv_list.append(id_gen_snv) # Se añaden todos los IDs en la lista vacía

# CNV:
for gen_cnv in cnv_list:
    if gen_cnv == 'RICTOR':
        id_cnv_list.append(253260)
        continue
    position = val_list.index(gen_cnv) # Número
    id_gen_cnv = (key_list[position]) # Número
    id_cnv_list.append(id_gen_cnv) # Se añaden todos los IDs en la lista vacía

""" Ahora se hace un bucle sobre la columna de mutaciones SNV y CNV del dataframe. Así, se busca en cada mutación de 
cada fila para ver en cuales de estas filas se encuentra el ID de los genes a estudiar. Se almacenan en una lista de 
listas los índices de las filas donde se encuentran los IDs de esos genes, de forma que se tiene una lista para cada gen. """
# SNV:
list_gen_snv = [[] for ID in range(len(id_snv_list))]

for index, id_snv in enumerate (id_snv_list): # Para cada ID del gen SNV de la lista...
    for index_row, row in enumerate (df_all_merge['SNV']): # Para cada fila dentro de la columna 'SNV'...
        for mutation in row: # Para cada mutación dentro de cada fila...
            if mutation[1] == id_snv: # Si el ID de la mutación es el mismo que el ID de la lista de genes...
                list_gen_snv[index].append(index_row) # Se almacena el índice de la fila en la lista de listas

# CNV:
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

""" Una vez se tienen almacenados los índices de las filas donde se producen las mutaciones SNV y CNV, hay que crear 
distintas columnas para cada uno de los genes objetivo, para asi mostrar la informacion de uno en uno. De esta forma, 
habra una columna distinta para cada gen SNV a estudiar; y tres columnas distintas para cada gen CNV a estudiar 
(amplificacion, delecion y estado normal). Ademas, se recopilan las columnas creadas en listas (una para las 
columnas de mutaciones SNV y otras tres para las columnas de mutaciones CNV). """
# SNV:
columns_list_snv = []

df_all_merge.drop(['SNV'], axis=1, inplace= True)
for gen_snv in snv_list:
    df_all_merge['SNV ' + gen_snv] = 0
    columns_list_snv.append('SNV ' + gen_snv)

# CNV:
columns_list_cnv_amp = []
columns_list_cnv_del = []

df_all_merge.drop(['CNV'], axis=1, inplace= True)
for gen_cnv in cnv_list:
    df_all_merge['CNV-A ' + gen_cnv] = 0
    df_all_merge['CNV-D ' + gen_cnv] = 0
    columns_list_cnv_amp.append('CNV-A ' + gen_cnv)
    columns_list_cnv_del.append('CNV-D ' + gen_cnv)

""" Una vez han sido creadas las columnas, se añade un '1' en aquellas filas donde el paciente tiene mutación sobre el
gen seleccionado. Se utiliza para ello los índices recogidos anteriormente en las respectivas listas de listas. De esta
forma, iterando sobre la lista de columnas creadas y mediante los distintos indices de cada sublista, se consigue
colocar un '1' en aquella filas donde el paciente tiene la mutacion especificada en el gen especificado. """
# SNV:
i_snv = 0
for column_snv in columns_list_snv:
    for index_snv_sublist in list_gen_snv[i_snv]:
        df_all_merge.loc[index_snv_sublist, column_snv] = 1
    i_snv += 1

# CNV:
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
df_all_merge.loc[df_all_merge.tumor_type == "Infiltrating Carcinoma (NOS)", "tumor_type"] = "Mixed Histology (NOS)"
df_all_merge.loc[df_all_merge.tumor_type == "Breast Invasive Carcinoma", "tumor_type"] = "Infiltrating Ductal Carcinoma"
df_all_merge.loc[df_all_merge.Neoadjuvant == "No", "Neoadjuvant"] = 0; df_all_merge.loc[df_all_merge.Neoadjuvant == "Yes", "Neoadjuvant"] = 1
df_all_merge.loc[df_all_merge.os_status == "0:LIVING", "os_status"] = 0; df_all_merge.loc[df_all_merge.os_status == "1:DECEASED", "os_status"] = 1
df_all_merge.loc[df_all_merge.dfs_status == "0:DiseaseFree", "dfs_status"] = 0; df_all_merge.loc[df_all_merge.dfs_status == "1:Recurred/Progressed", "dfs_status"] = 1

# Cambiar los subtipos para que muestre solo los receptores de HER-2
df_all_merge.loc[df_all_merge.subtype == "BRCA_Her2", "subtype"] = 1
df_all_merge['subtype'].replace({"BRCA_Normal": 0, "BRCA_Basal": 0, "BRCA_LumA": 0, "BRCA_LumB": 0}, inplace = True)

""" Ahora se procede a procesar la columna continua de edad, que se normaliza para que esté en el rango de (0-1) """
scaler = MinMaxScaler()
train_continuous = scaler.fit_transform(df_all_merge[['Age']])
df_all_merge.loc[:, 'Age'] = train_continuous[:, 0]

""" Ahora se eliminan las filas donde haya datos nulos para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace = True)

""" Se eliminan las columnas de TNM (ya se tiene la columna STAGE que da la misma información) """
df_all_merge = df_all_merge.drop(columns = ['path_t_stage', 'path_n_stage', 'path_m_stage'])

""" Una vez la tabla tiene las columnas deseadas se procede a codificar las columnas categóricas del dataframe a valores
numéricos mediante la técnica del 'One Hot Encoding'. Más adelante se escalarán las columnas numéricas continuas, pero
ahora se realiza esta técnica antes de hacer la repartición de subconjuntos para que no haya problemas con las columnas. """
#@ get_dummies: Aplica técnica de 'One Hot Encoding', creando columnas binarias para las columnas seleccionadas
df_all_merge = pd.get_dummies(df_all_merge, columns=["tumor_type", "stage"])

""" Se reordenan las columnas del dataframe para colocar las nuevas columnas numericas de datos anatomopatológicos antes
de las mutaciones """
cols = df_all_merge.columns.tolist()
cols = cols[:6] + cols[-15:] + cols[6:-15]
df_all_merge = df_all_merge[cols]

""" Se renombran algunas columnas, simplemente para hacerlo más atractivo visualmente. """
df_all_merge = df_all_merge.rename(columns = {'subtype': 'HER-2 receptor', 'tumor_type_Infiltrating Ductal Carcinoma': 'Ductal [tumor type]',
                                              'os_status': 'Survival', 'dfs_status': 'Relapse',
                                              'tumor_type_Infiltrating Lobular Carcinoma': 'Lobular [tumor type]',
                                              'tumor_type_Medullary Carcinoma': 'Medullary [tumor type]',
                                              'tumor_type_Metaplastic Carcinoma': 'Metaplastic [tumor type]',
                                              'tumor_type_Mixed Histology (NOS)': 'Mixed [tumor type]',
                                              'tumor_type_Mucinous Carcinoma': 'Mucinous [tumor type]',
                                              'tumor_type_Other': 'Other [tumor type]', 'stage_STAGE IB': 'STAGE IB',
                                              'stage_STAGE II': 'STAGE II', 'stage_STAGE IIA': 'STAGE IIA',
                                              'stage_STAGE IIB': 'STAGE IIB', 'stage_STAGE III': 'STAGE III',
                                              'stage_STAGE IIIA': 'STAGE IIIA', 'stage_STAGE IIIB': 'STAGE IIIB',
                                              'stage_STAGE IIIC': 'STAGE IIIC', 'stage_STAGE IV': 'STAGE IV',
                                              'stage_STAGE X': 'STAGE X'})

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
train_data, test_data = train_test_split(df_all_merge, test_size = 0.20, stratify = df_all_merge['SNV PIK3CA'])
train_data, valid_data = train_test_split(train_data, test_size = 0.15, stratify = train_data['SNV PIK3CA'])

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
train_data.dropna(inplace = True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.
valid_data.dropna(inplace = True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.
test_data.dropna(inplace = True)  # Mantiene el DataFrame con las entradas válidas en la misma variable.

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

""" Se iguala el número de teselas con mutación y sin mutación SNV del gen PIK3CA """
# Entrenamiento
train_pik3ca_tiles = train_data['SNV PIK3CA'].value_counts()[1] # Con mutación
train_no_pik3ca_tiles = train_data['SNV PIK3CA'].value_counts()[0] # Sin mutación

if train_no_pik3ca_tiles >= train_pik3ca_tiles:
    difference_train = train_no_pik3ca_tiles - train_pik3ca_tiles
    train_data = train_data.sort_values(by = 'SNV PIK3CA', ascending = False)
else:
    difference_train = train_pik3ca_tiles - train_no_pik3ca_tiles
    train_data = train_data.sort_values(by = 'SNV PIK3CA', ascending = True)

train_data = train_data[:-difference_train]

# Validación
valid_pik3ca_tiles = valid_data['SNV PIK3CA'].value_counts()[1] # Con mutación
valid_no_pik3ca_tiles = valid_data['SNV PIK3CA'].value_counts()[0] # Sin mutación

if valid_no_pik3ca_tiles >= valid_pik3ca_tiles:
    difference_valid = valid_no_pik3ca_tiles - valid_pik3ca_tiles
    valid_data = valid_data.sort_values(by = 'SNV PIK3CA', ascending = False)
else:
    difference_valid = valid_pik3ca_tiles - valid_no_pik3ca_tiles
    valid_data = valid_data.sort_values(by = 'SNV PIK3CA', ascending = True)

valid_data = valid_data[:-difference_valid]

# Test
test_pik3ca_tiles = test_data['SNV PIK3CA'].value_counts()[1] # Con mutación
test_no_pik3ca_tiles = test_data['SNV PIK3CA'].value_counts()[0] # Sin mutación

if test_no_pik3ca_tiles >= test_pik3ca_tiles:
    difference_test = test_no_pik3ca_tiles - test_pik3ca_tiles
    test_data = test_data.sort_values(by = 'SNV PIK3CA', ascending = False)
else:
    difference_test = test_pik3ca_tiles - test_no_pik3ca_tiles
    test_data = test_data.sort_values(by = 'SNV PIK3CA', ascending = True)

test_data = test_data[:-difference_test]

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
train_data = train_data.drop(['img_path', 'ID'], axis = 1)
valid_data = valid_data.drop(['img_path', 'ID'], axis = 1)
test_data = test_data.drop(['img_path', 'ID'], axis = 1)

""" Se extraen los datos de salida de la red neuronal y se guardan los nombres de las columnas de datos """
train_labels_pik3ca = train_data.pop('SNV PIK3CA')
valid_labels_pik3ca = valid_data.pop('SNV PIK3CA')
test_labels_pik3ca = test_data.pop('SNV PIK3CA')

train_data_columns = train_data.columns.values

""" Para poder entrenar la red hace falta transformar las tablas en arrays. Para ello se utiliza 'numpy' """
train_data = np.asarray(train_data).astype('float32')
valid_data = np.asarray(valid_data).astype('float32')
test_data = np.asarray(test_data).astype('float32')

train_labels_pik3ca = np.asarray(train_labels_pik3ca).astype('float32')
valid_labels_pik3ca = np.asarray(valid_labels_pik3ca).astype('float32')
test_labels_pik3ca = np.asarray(test_labels_pik3ca).astype('float32')

""" Se borran los dataframes utilizados, puesto que ya no sirven para nada, y se recopila la longitud de las imágenes de
entrenamiento y validacion para utilizarlas posteriormente en el entrenamiento: """
del df_path_n_stage, df_list

train_image_data_len = len(train_image_data)
valid_image_data_len = len(valid_image_data)
batch_dimension = 32

""" Se pueden guardar en formato de 'numpy' las imágenes y las etiquetas de test para usarlas después de entrenar la red
neuronal convolucional. """
#np.save('test_data', test_data)
#np.save('test_image', test_image_data)
#np.save('test_labels', test_labels_pik3ca)

""" Se mide la importancia de las variables de datos con Random Forest. Se crean grupos de árboles de decisión para
estimar cuales son las variables que mas influyen en la predicción de la salida"""
feature_names = [f"feature {i}" for i in range(train_data.shape[1])]
forest = RandomForestClassifier(random_state = 0)
forest.fit(train_data, train_labels_pik3ca)

result = permutation_importance(forest, train_data, train_labels_pik3ca, n_repeats = 30, random_state = 42,
                                n_jobs = 2)
forest_importances = pd.Series(result.importances_mean, index = train_data_columns)
forest_importances_threshold = forest_importances.nlargest(n = 10).dropna()

fig, ax = plt.subplots()
forest_importances_threshold.plot.barh(ax = ax)
ax.set_title("Importancia de variables [SNV PIK3CA]")
ax.set_ylabel("Reducción de eficacia media")
fig.tight_layout()
plt.show()

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL CONVOLUCIONAL ---------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
""" Se definen dos entradas para dos ramas distintas: una para la red perceptrón multicapa; y otra para la red neuronal 
convolucional """
input_data = Input(shape = (train_data.shape[1]),)
input_image = Input(shape = (alto, ancho, canales))

""" La primera rama del modelo (Perceptrón multicapa) opera con la entrada de datos: """
mlp = layers.Dense(train_data.shape[1], activation = "relu")(input_data)
mlp = layers.Dropout(0.2)(mlp)
mlp = layers.Dense(128, activation = "relu")(mlp)
mlp = layers.Dropout(0.2)(mlp)
mlp_out = layers.Dense(64, activation = "relu")(mlp)

""" La segunda rama del modelo será la encargada de procesar las imágenes: """
cnn_model = keras.applications.EfficientNetB7(weights = 'imagenet', input_tensor = input_image,include_top = False,
                                              pooling = 'max')

""" Se añaden capas de clasificación después de las capas congeladas de convolución. """
all_cnn_model = cnn_model.output
all_cnn_model = layers.Flatten()(all_cnn_model)
all_cnn_model = layers.Dense(512, activation = "relu")(all_cnn_model)
all_cnn_model = layers.Dropout(0.2)(all_cnn_model)
all_cnn_model = layers.Dense(128, activation = "relu")(all_cnn_model)
all_cnn_model = layers.Dropout(0.2)(all_cnn_model)
all_cnn_model_out = layers.Dense(64, activation = "relu")(all_cnn_model)

""" Se congelan todas las capas convolucionales del modelo base de la red convolucional. """
# A partir de TF 2.0 @trainable = False hace tambien ejecutar las capas BN en modo inferencia (@training = False)
for layer in cnn_model.layers:
    layer.trainable = False

""" Se combina la salida de ambas ramas. """
combined = keras.layers.concatenate([mlp_out, all_cnn_model_out])

""" Una vez se ha concatenado la salida de ambas ramas, se aplica dos capas densamente conectadas, la última de ellas
siendo la de la predicción final con activación 'sigmoid', puesto que la salida será binaria. """
multi_input_model = layers.Dense(64, activation = "relu")(combined)
multi_input_model = layers.Dropout(0.2)(multi_input_model)
multi_input_model = layers.Dense(16, activation = "relu")(multi_input_model)
multi_input_model = layers.Dropout(0.2)(multi_input_model)
multi_input_model_out = layers.Dense(1, activation = "sigmoid")(multi_input_model)

""" El modelo final aceptará datos numéricos/categóricos en la entrada de la red perceptrón multicapa e imágenes en la
red neuronal convolucional, de forma que a la salida solo se obtenga la predicción de la metástasis a distancia. """
model = keras.models.Model(inputs = [input_data, input_image], outputs = multi_input_model_out)

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

model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(learning_rate = 0.00001),
              metrics = metrics)
model.summary()

""" Se implementa un callbacks para guardar el modelo cada época. """
checkpoint_path = '/home/avalderas/img_slides/mutations/image&data/PIK3CA SNV/inference/models/model_image&data_pik3ca_{epoch:02d}_{val_loss:.2f}.h5'
mcp_loss = ModelCheckpoint(filepath = checkpoint_path, monitor = 'val_loss', mode = 'min', save_best_only = True)
mcp_accuracy = ModelCheckpoint(filepath = checkpoint_path, monitor = 'val_accuracy', mode = 'max', save_best_only = True)

""" Una vez definido el modelo, se entrena: """
model.fit(x = [train_data, train_image_data], y = train_labels_pik3ca, epochs = 2, verbose = 1,
          validation_data = ([valid_data, valid_image_data], valid_labels_pik3ca),
          steps_per_epoch = (train_image_data_len / batch_dimension),
          validation_steps = (valid_image_data_len / batch_dimension))

""" Una vez el modelo ya ha sido entrenado, se descongelan algunas capas convolucionales del modelo base de la red para 
reeentrenar el modelo ('fine tuning'). Este es un último paso opcional que puede dar grandes mejoras o un rápido 
sobreentrenamiento y que solo debe ser realizado después de entrenar el modelo con las capas congeladas """
set_trainable = 0

for layer in cnn_model.layers:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

""" Es importante recompilar el modelo después de hacer cualquier cambio al atributo 'trainable', para que los cambios
se tomen en cuenta. """
model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.00001),
              loss = 'binary_crossentropy',
              metrics = metrics)
model.summary()

""" Una vez descongelado las capas convolucionales seleccionadas y compilado de nuevo el modelo, se entrena otra vez. """
neural_network = model.fit(x = [train_data, train_image_data], y = train_labels_pik3ca, epochs = 20, verbose = 1,
                           validation_data = ([valid_data, valid_image_data], valid_labels_pik3ca),
                           #callbacks = [mcp_loss, mcp_accuracy],
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
plt.figure() # Crea o activa una figura