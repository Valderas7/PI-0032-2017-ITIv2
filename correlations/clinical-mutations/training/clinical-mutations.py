import shelve # datos persistentes
import pandas as pd
import numpy as np
import imblearn
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
import itertools
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import * # Para instanciar tensores de Keras
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.model_selection import train_test_split # Se importa la librería para dividir los datos en entreno y test.
from sklearn.preprocessing import MinMaxScaler # Para escalar valores
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix # Para realizar la matriz de confusión

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN DATOS TABULARES ------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# 1) Datos clínicos: Edad diagnóstico, neoadyuvante, antecedentes, metástasis a distancia, supervivencia, recaída.
# 2) Datos de mutaciones: SNV, CNV-A, CNV-D.
list_to_read = ['CNV_oncomine', 'age', 'all_oncomine', 'mutations_oncomine', 'cancer_type', 'cancer_type_detailed',
                'dfs_months', 'dfs_status', 'dict_genes', 'dss_months', 'dss_status', 'ethnicity',
                'full_length_oncomine', 'fusions_oncomine', 'muted_genes', 'CNA_genes', 'hotspot_oncomine', 'mutations',
                'CNAs', 'neoadjuvant', 'os_months', 'os_status', 'path_m_stage', 'path_n_stage', 'path_t_stage', 'sex',
                'stage', 'subtype', 'tumor_type', 'new_tumor', 'person_neoplasm_status', 'prior_diagnosis',
                'pfs_months', 'pfs_status', 'radiation_therapy']

filename = '/home/avalderas/img_slides/data/brca_tcga_pan_can_atlas_2018.out'
#filename = 'C:\\Users\\valde\Desktop\Datos_repositorio\\tcga_data\data/brca_tcga_pan_can_atlas_2018.out'

""" Almacenamos en una variable los diccionarios: """
with shelve.open(filename) as data:
    dict_genes = data.get('dict_genes')
    age = data.get('age')
    cancer_type_detailed = data.get('cancer_type_detailed')
    neoadjuvant = data.get('neoadjuvant')  # 1 valor nulo
    os_months = data.get('os_months')
    os_status = data.get('os_status')
    path_m_stage = data.get('path_m_stage')
    path_n_stage = data.get('path_n_stage')
    path_t_stage = data.get('path_t_stage')
    stage = data.get('stage')
    subtype = data.get('subtype')
    tumor_type = data.get('tumor_type')
    new_tumor = data.get('new_tumor')  # ~200 valores nulos
    prior_diagnosis = data.get('prior_diagnosis')  # 1 valor nulo
    pfs_months = data.get('pfs_months')  # 2 valores nulos
    pfs_status = data.get('pfs_status')  # 1 valor nulo
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
df_dfs_status = pd.DataFrame.from_dict(dfs_status.items()); df_dfs_status.rename(columns = {0 : 'ID', 1 : 'dfs_status'}, inplace = True)
df_path_n_stage = pd.DataFrame.from_dict(path_n_stage.items()); df_path_n_stage.rename(columns = {0 : 'ID', 1 : 'path_n_stage'}, inplace = True)
df_path_t_stage = pd.DataFrame.from_dict(path_t_stage.items()); df_path_t_stage.rename(columns = {0 : 'ID', 1 : 'path_t_stage'}, inplace = True)
df_stage = pd.DataFrame.from_dict(stage.items()); df_stage.rename(columns = {0 : 'ID', 1 : 'stage'}, inplace = True)
df_subtype = pd.DataFrame.from_dict(subtype.items()); df_subtype.rename(columns = {0 : 'ID', 1 : 'subtype'}, inplace = True)
df_tumor_type = pd.DataFrame.from_dict(tumor_type.items()); df_tumor_type.rename(columns = {0 : 'ID', 1 : 'tumor_type'}, inplace = True)
df_prior_diagnosis = pd.DataFrame.from_dict(prior_diagnosis.items()); df_prior_diagnosis.rename(columns = {0 : 'ID', 1 : 'prior_diagnosis'}, inplace = True)
df_snv = pd.DataFrame.from_dict(snv.items()); df_snv.rename(columns = {0 : 'ID', 1 : 'SNV'}, inplace = True)
df_cnv = pd.DataFrame.from_dict(cnv.items()); df_cnv.rename(columns = {0 : 'ID', 1 : 'CNV'}, inplace = True)
df_os_status = pd.DataFrame.from_dict(os_status.items()); df_os_status.rename(columns = {0 : 'ID', 1 : 'os_status'}, inplace = True)

df_path_m_stage = pd.DataFrame.from_dict(path_m_stage.items()); df_path_m_stage.rename(columns = {0 : 'ID', 1 : 'path_m_stage'}, inplace = True)

df_list = [df_age, df_neoadjuvant, df_prior_diagnosis, df_os_status, df_dfs_status, df_path_m_stage, df_path_n_stage,
           df_snv, df_cnv]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

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

""" Una vez creada esta columna y habiendo asignado los valores correspondientes, se elimina la columna del estadio 'M'
que ya no es necesario y no es un dato clínico."""
df_all_merge = df_all_merge.drop('path_m_stage', 1)

""" Ahora se va a encontrar cuales son los ID de los genes que nos interesa. Para empezar se carga el archivo excel 
donde aparecen todos los genes con mutaciones que interesan estudiar usando 'openpyxl' y creamos dos listas. Una para
los genes SNV y otra para los genes CNV."""
mutations_target = pd.read_excel('/home/avalderas/img_slides/excel_genesOCA&inibica_patients/Panel_OCA.xlsx',
                                 usecols= 'B:C', engine= 'openpyxl')
#mutations_target = pd.read_excel('C:\\Users\\valde\Desktop\Datos_repositorio\\img_slides\excel_oca_genes/Panel_OCA.xlsx',
                                 #usecols= 'B:C', engine= 'openpyxl')

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
    df_all_merge['SNV_' + gen_snv] = 0
    columns_list_snv.append('SNV_' + gen_snv)

# CNV:
columns_list_cnv_amp = []
columns_list_cnv_del = []

df_all_merge.drop(['CNV'], axis=1, inplace= True)
for gen_cnv in cnv_list:
    df_all_merge['CNV_' + gen_cnv + '_AMP'] = 0
    df_all_merge['CNV_' + gen_cnv + '_DEL'] = 0
    columns_list_cnv_amp.append('CNV_' + gen_cnv + '_AMP')
    columns_list_cnv_del.append('CNV_' + gen_cnv + '_DEL')

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

""" En este caso, se eliminan los pacientes con categoria 'N0 o 'NX', aquellos pacientes a los que no se les puede 
determinar si tienen o no metastasis. """
df_all_merge = df_all_merge[(df_all_merge["path_n_stage"]!='N0') & (df_all_merge["path_n_stage"]!='NX') &
                            (df_all_merge["path_n_stage"]!='N0 (I-)') & (df_all_merge["path_n_stage"]!='N0 (I+)') &
                            (df_all_merge["path_n_stage"]!='N0 (MOL+)')]

""" Una vez eliminados los pacientes con N0 o NX se elimina la columna del estadio N puesto que no es un dato clínico. """
df_all_merge = df_all_merge.drop('path_n_stage', 1)

""" Se convierten las columnas de pocos valores en columnas binarias: """
df_all_merge.loc[df_all_merge.neoadjuvant == "No", "neoadjuvant"] = 0; df_all_merge.loc[df_all_merge.neoadjuvant == "Yes", "neoadjuvant"] = 1
df_all_merge.loc[df_all_merge.prior_diagnosis == "No", "prior_diagnosis"] = 0; df_all_merge.loc[df_all_merge.prior_diagnosis == "Yes", "prior_diagnosis"] = 1
df_all_merge.loc[df_all_merge.os_status == "0:LIVING", "os_status"] = 0; df_all_merge.loc[df_all_merge.os_status == "1:DECEASED", "os_status"] = 1
df_all_merge.loc[df_all_merge.dfs_status == "0:DiseaseFree", "dfs_status"] = 0; df_all_merge.loc[df_all_merge.dfs_status == "1:Recurred/Progressed", "dfs_status"] = 1

""" Ahora, antes de transformar las variables categóricas en numéricas, se eliminan las filas donde haya datos nulos
para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
# 337train, 38val, 94test
train_tabular_data, test_tabular_data = train_test_split(df_all_merge, test_size = 0.20)
train_tabular_data, valid_tabular_data = train_test_split(train_tabular_data, test_size = 0.10)

""" Ya se puede eliminar de los dos subconjuntos la columna 'ID' que no es útil para la red MLP: """
train_tabular_data = train_tabular_data.drop(['ID'], axis=1)
valid_tabular_data = valid_tabular_data.drop(['ID'], axis=1)
test_tabular_data = test_tabular_data.drop(['ID'], axis=1)

""" Se dividen los datos clínicos y los datos de anatomía patológica en datos de entrada y datos de salida."""
train_labels_snv = train_tabular_data.iloc[:, 6:157]
valid_labels_snv = valid_tabular_data.iloc[:, 6:157]
test_labels_snv = test_tabular_data.iloc[:, 6:157]

train_labels_cnv_a = train_tabular_data.iloc[:, 157::2]
valid_labels_cnv_a = valid_tabular_data.iloc[:, 157::2]
test_labels_cnv_a = test_tabular_data.iloc[:, 157::2]

train_labels_cnv_d = train_tabular_data.iloc[:, 158::2]
valid_labels_cnv_d = valid_tabular_data.iloc[:, 158::2]
test_labels_cnv_d = test_tabular_data.iloc[:, 158::2]

train_tabular_data = train_tabular_data.iloc[:, :6]
valid_tabular_data = valid_tabular_data.iloc[:, :6]
test_tabular_data = test_tabular_data.iloc[:, :6]

""" Se extraen los nombres de las columnas de las clases de salida para usarlos posteriormente en la sección de
evaluación: """
test_columns_snv = test_labels_snv.columns.values
classes_snv = test_columns_snv.tolist()

test_columns_cnv_a = test_labels_cnv_a.columns.values
classes_cnv_a = test_columns_cnv_a.tolist()

test_columns_cnv_d = test_labels_cnv_d.columns.values
classes_cnv_d = test_columns_cnv_d.tolist()

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

""" Para poder entrenar la red hace falta transformar los dataframes de entrenamiento y test en arrays de numpy, así 
como también la columna de salida de ambos subconjuntos (las imágenes YA fueron convertidas anteriormente, por lo que no
hace falta transformarlas de nuevo). """
train_tabular_data = np.asarray(train_tabular_data).astype('float32')
train_labels_snv = np.asarray(train_labels_snv).astype('float32')
train_labels_cnv_a = np.asarray(train_labels_cnv_a).astype('float32')
train_labels_cnv_d = np.asarray(train_labels_cnv_d).astype('float32')

valid_tabular_data = np.asarray(valid_tabular_data).astype('float32')
valid_labels_snv = np.asarray(valid_labels_snv).astype('float32')
valid_labels_cnv_a = np.asarray(valid_labels_cnv_a).astype('float32')
valid_labels_cnv_d = np.asarray(valid_labels_cnv_d).astype('float32')

test_tabular_data = np.asarray(test_tabular_data).astype('float32')
test_labels_snv = np.asarray(test_labels_snv).astype('float32')
test_labels_cnv_a = np.asarray(test_labels_cnv_a).astype('float32')
test_labels_cnv_d = np.asarray(test_labels_cnv_d).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (MLP) -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
Input_ = Input(shape=train_tabular_data.shape[1], )
model = layers.Dense(128, activation= 'relu')(Input_)
model = layers.Dropout(0.5)(model)
model = layers.Dense(64, activation= 'relu')(model)
model = layers.Dropout(0.5)(model)
output1 = layers.Dense(train_labels_snv.shape[1], activation = "sigmoid", name= 'snv')(model)
output2 = layers.Dense(train_labels_cnv_a.shape[1], activation = "sigmoid", name = 'cnv_a')(model)
output3 = layers.Dense(train_labels_cnv_d.shape[1], activation = "sigmoid", name= 'cnv_d')(model)

model = Model(inputs = Input_, outputs = [output1, output2, output3])

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

model.compile(loss = {'snv': 'binary_crossentropy', 'cnv_a': 'binary_crossentropy', 'cnv_d': 'binary_crossentropy'},
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)

model.summary()

""" Se implementa un callback: para guardar el mejor modelo que tenga la menor 'loss' en la validación. """
checkpoint_path = '/correlations/clinical-mutations/inference/test_data/clinical-mutations.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = True, monitor= 'loss', mode= 'min')

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
neural_network = model.fit(x = train_tabular_data,
                           y = {'snv': train_labels_snv, 'cnv_a': train_labels_cnv_a, 'cnv_d': train_labels_cnv_d},
                           epochs = 1000,
                           verbose = 1,
                           batch_size = 32,
                           #callbacks= mcp_save,
                           validation_data = (valid_tabular_data, {'snv': valid_labels_snv, 'cnv_a': valid_labels_cnv_a,
                                                                   'cnv_d': valid_labels_cnv_d}))

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_tabular_data, [test_labels_snv, test_labels_cnv_a, test_labels_cnv_d], verbose = 0)

print("\n'Loss' de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "SNV del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones SNV del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f} \n"
      "Exactitud de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones SNV"
      " del panel OCA en el conjunto de prueba: {:.2f}".format(results[1], results[8], results[9],
                                                               results[6]/(results[6]+results[5]), results[10] * 100,
                                                               results[11]))
if results[8] > 0 or results[9] > 0:
    print("Valor-F de las mutaciones SNV del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[8] * results[9]) /
                                                                                                (results[8] + results[9])))

print("\n'Loss' de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones CNV-A del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}\n"
      "Exactitud de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones "
      "CNV-A del panel OCA en el conjunto de prueba: {:.2f}".format(results[2], results[16], results[17],
                                                                    results[14]/(results[14]+results[13]),
                                                                    results[18] * 100, results[19]))
if results[16] > 0 or results[17] > 0:
    print("Valor-F de las mutaciones CNV-A del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[16] * results[17]) /
                                                                                                  (results[16] + results[17])))

print("\n'Loss' de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""Sensibilidad de las mutaciones "
      "CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n""Precisión de las mutaciones CNV-D del panel OCA en el "
      "conjunto de prueba: {:.2f}\n""Especifidad de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}\n"
      "Exactitud de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f} %\n""AUC-ROC de las mutaciones "
      "CNV-D del panel OCA en el conjunto de prueba: {:.2f}".format(results[3], results[24], results[25],
                                                                    results[22]/(results[22]+results[21]),
                                                                    results[26] * 100, results[27]))
if results[24] > 0 or results[25] > 0:
    print("Valor-F de las mutaciones CNV-D del panel OCA en el conjunto de prueba: {:.2f}".format((2 * results[24] * results[25]) /
                                                                                                  (results[24] + results[25])))

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
# @suppress = True: Muestra los números con representación de coma fija
# @predict: Genera predicciones para nuevas entradas
#print("\nGenera predicciones para 10 muestras")
#print("Clase del primer paciente: \n", test_labels_snv[:1], test_labels_cnv_a[:1], test_labels_cnv_d[:1])
np.set_printoptions(precision=3, suppress=True)
#print("\nPredicciones:\n", np.round(model.predict(test_tabular_data[:1])[0])) # El índice da una salida u otra

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
# SNV
y_true_snv = test_labels_snv
y_pred_snv = np.round(model.predict(test_tabular_data)[0])

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
y_pred_cnv_a = np.round(model.predict(test_tabular_data)[1])

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
y_pred_cnv_d = np.round(model.predict(test_tabular_data)[2])

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

#np.save('test_data', test_tabular_data)
#np.save('test_labels_snv', test_labels_snv)
#np.save('test_labels_cnv_a', test_labels_cnv_a)
#np.save('test_labels_cnv_d', test_labels_cnv_d)