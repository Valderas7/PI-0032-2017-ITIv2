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
from tensorflow.keras.layers import *
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------------- SECCIÓN DATOS TABULARES ------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# 1) Datos de mutaciones: SNV, CNV-A, CNV-D.
# 2) Datos anatomopatológicos: Tipo histológico, STAGE, pT, pN, pM, IHQ.
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

df_list = [df_tumor_type, df_stage, df_path_t_stage, df_path_n_stage, df_path_m_stage, df_subtype, df_snv, df_cnv]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

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

""" Se convierten las columnas de pocos valores en columnas binarias: """
df_all_merge.loc[df_all_merge.tumor_type == "Infiltrating Carcinoma (NOS)", "tumor_type"] = "Mixed Histology (NOS)"
df_all_merge.loc[df_all_merge.tumor_type == "Breast Invasive Carcinoma", "tumor_type"] = "Infiltrating Ductal Carcinoma"
df_all_merge.loc[df_all_merge.path_m_stage == "CM0 (I+)", "path_m_stage"] = "M0"

""" Ahora, antes de transformar las variables categóricas en numéricas, se eliminan las filas donde haya datos nulos
para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Una vez la tabla tiene las columnas deseadas se procede a codificar las columnas categóricas del dataframe a valores
numéricos mediante la técnica del 'One Hot Encoding'. Más adelante se escalarán las columnas numéricas continuas, pero
ahora se realiza esta técnica antes de hacer la repartición de subconjuntos para que no haya problemas con las columnas. """
#@ get_dummies: Aplica técnica de 'One Hot Encoding', creando columnas binarias para las columnas seleccionadas
df_all_merge = pd.get_dummies(df_all_merge, columns=["tumor_type", "stage", "path_t_stage", "path_n_stage",
                                                     "path_m_stage", "subtype"])

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
# 352train, 40val, 98test
train_tabular_data, test_tabular_data = train_test_split(df_all_merge, test_size = 0.20)
train_tabular_data, valid_tabular_data = train_test_split(train_tabular_data, test_size = 0.10)

""" Ya se puede eliminar de los dos subconjuntos la columna 'ID' que no es útil para la red MLP: """
train_tabular_data = train_tabular_data.drop(['ID'], axis = 1)
valid_tabular_data = valid_tabular_data.drop(['ID'], axis = 1)
test_tabular_data = test_tabular_data.drop(['ID'], axis = 1)

""" Se dividen los datos de mutaciones y de anatomía patológica en datos de entrada y datos de salida, respectivamente """
train_labels_tumor_type = train_tabular_data.iloc[:, 237:244]
valid_labels_tumor_type = valid_tabular_data.iloc[:, 237:244]
test_labels_tumor_type = test_tabular_data.iloc[:, 237:244]

train_labels_STAGE = train_tabular_data.iloc[:, 244:254]
valid_labels_STAGE = valid_tabular_data.iloc[:, 244:254]
test_labels_STAGE = test_tabular_data.iloc[:, 244:254]

train_labels_pT = train_tabular_data.iloc[:, 254:264]
valid_labels_pT = valid_tabular_data.iloc[:, 254:264]
test_labels_pT = test_tabular_data.iloc[:, 254:264]

train_labels_pN = train_tabular_data.iloc[:, 264:275]
valid_labels_pN = valid_tabular_data.iloc[:, 264:275]
test_labels_pN = test_tabular_data.iloc[:, 264:275]

train_labels_pM = train_tabular_data.iloc[:, 275:278]
valid_labels_pM = valid_tabular_data.iloc[:, 275:278]
test_labels_pM = test_tabular_data.iloc[:, 275:278]

train_labels_IHQ = train_tabular_data.iloc[:, 278:]
valid_labels_IHQ = valid_tabular_data.iloc[:, 278:]
test_labels_IHQ = test_tabular_data.iloc[:, 278:]

train_tabular_data = train_tabular_data.iloc[:, :237]
valid_tabular_data = valid_tabular_data.iloc[:, :237]
test_tabular_data = test_tabular_data.iloc[:, :237]

""" Para poder entrenar la red hace falta transformar los dataframes en arrays de numpy. """
train_tabular_data = np.asarray(train_tabular_data).astype('float32')

train_labels_tumor_type = np.asarray(train_labels_tumor_type).astype('float32')
train_labels_STAGE = np.asarray(train_labels_STAGE).astype('float32')
train_labels_pT = np.asarray(train_labels_pT).astype('float32')
train_labels_pN = np.asarray(train_labels_pN).astype('float32')
train_labels_pM = np.asarray(train_labels_pM).astype('float32')
train_labels_IHQ = np.asarray(train_labels_IHQ).astype('float32')

valid_tabular_data = np.asarray(valid_tabular_data).astype('float32')

valid_labels_tumor_type = np.asarray(valid_labels_tumor_type).astype('float32')
valid_labels_STAGE = np.asarray(valid_labels_STAGE).astype('float32')
valid_labels_pT = np.asarray(valid_labels_pT).astype('float32')
valid_labels_pN = np.asarray(valid_labels_pN).astype('float32')
valid_labels_pM = np.asarray(valid_labels_pM).astype('float32')
valid_labels_IHQ = np.asarray(valid_labels_IHQ).astype('float32')

test_tabular_data = np.asarray(test_tabular_data).astype('float32')

test_labels_tumor_type = np.asarray(test_labels_tumor_type).astype('float32')
test_labels_STAGE = np.asarray(test_labels_STAGE).astype('float32')
test_labels_pT = np.asarray(test_labels_pT).astype('float32')
test_labels_pN = np.asarray(test_labels_pN).astype('float32')
test_labels_pM = np.asarray(test_labels_pM).astype('float32')
test_labels_IHQ = np.asarray(test_labels_IHQ).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (MLP) -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
Input_ = Input(shape=train_tabular_data.shape[1], )
model = layers.Dense(128, activation= 'relu')(Input_)
model = layers.Dropout(0.5)(model)
model = layers.Dense(64, activation= 'relu')(model)
model = layers.Dropout(0.5)(model)
output1 = layers.Dense(train_labels_tumor_type.shape[1], activation = "softmax", name= 'tumor_type')(model)
output2 = layers.Dense(train_labels_STAGE.shape[1], activation = "softmax", name = 'STAGE')(model)
output3 = layers.Dense(train_labels_pT.shape[1], activation = "softmax", name= 'pT')(model)
output4 = layers.Dense(train_labels_pN.shape[1], activation = "softmax", name= 'pN')(model)
output5 = layers.Dense(train_labels_pM.shape[1], activation = "softmax", name= 'pM')(model)
output6 = layers.Dense(train_labels_IHQ.shape[1], activation = "softmax", name= 'IHQ')(model)

model = Model(inputs = Input_, outputs = [output1, output2, output3, output4, output5, output6])

""" Hay que definir las métricas de la red y configurar los distintos hiperparámetros para entrenar la red. El modelo ya
ha sido definido anteriormente, así que ahora hay que compilarlo. Para ello se define una función de loss y un 
optimizador. Con la función de loss se estimará la 'loss' del modelo. Por su parte, el optimizador actualizará los
parámetros de la red neuronal con el objetivo de minimizar la función de 'loss'. """
# @lr: tamaño de pasos para alcanzar el mínimo global de la función de loss.
metrics = [keras.metrics.TruePositives(name = 'tp'), keras.metrics.FalsePositives(name = 'fp'),
           keras.metrics.TrueNegatives(name = 'tn'), keras.metrics.FalseNegatives(name = 'fn'),
           keras.metrics.Recall(name = 'recall'), # TP / (TP + FN)
           keras.metrics.Precision(name = 'precision'), # TP / (TP + FP)
           keras.metrics.BinaryAccuracy(name = 'accuracy'), keras.metrics.AUC(name = 'AUC')]

model.compile(loss = {'tumor_type': 'categorical_crossentropy', 'STAGE': 'categorical_crossentropy',
                      'pT': 'categorical_crossentropy', 'pN': 'categorical_crossentropy',
                      'pM': 'categorical_crossentropy', 'IHQ': 'categorical_crossentropy'},
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)

model.summary()

""" Se implementa un callback: para guardar el mejor modelo que tenga la menor 'loss' en la validación. """
checkpoint_path = '/correlations/mutations-anatomopathologic/inference/test_data/mutations-anatomopathologic.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = True, monitor= 'val_loss', mode= 'min')

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
neural_network = model.fit(x = train_tabular_data,
                           y = {'tumor_type': train_labels_tumor_type, 'STAGE': train_labels_STAGE,
                                'pT': train_labels_pT, 'pN': train_labels_pN, 'pM': train_labels_pM,
                                'IHQ': train_labels_IHQ},
                           epochs = 1000,
                           verbose = 1,
                           batch_size = 32,
                           #callbacks= mcp_save,
                           validation_data = (valid_tabular_data, {'tumor_type': valid_labels_tumor_type,
                                                                   'STAGE': valid_labels_STAGE, 'pT': valid_labels_pT,
                                                                   'pN': valid_labels_pN, 'pM': valid_labels_pM,
                                                                   'IHQ': valid_labels_IHQ}))


""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_tabular_data, [test_labels_tumor_type, test_labels_STAGE, test_labels_pT, test_labels_pN,
                                             test_labels_pM, test_labels_IHQ], verbose = 0)

print("\n'Loss' del tipo histológico en el conjunto de prueba: {:.2f}\n""Sensibilidad del tipo histológico en el "
      "conjunto de prueba: {:.2f}\n""Precisión del tipo histológico en el conjunto de prueba: {:.2f}\n""Especifidad del "
      "tipo histológico en el conjunto de prueba: {:.2f} \n""Exactitud del tipo histológico en el conjunto de prueba: "
      "{:.2f} %\n""AUC-ROC del tipo histológico en el conjunto de prueba: {:.2f}".format(results[1], results[11],
                                                                                           results[12],
                                                                                           results[9]/(results[9]+results[8]),
                                                                                           results[13] * 100, results[14]))
if results[11] > 0 or results[12] > 0:
    print("Valor-F del tipo histológico en el conjunto de prueba: {:.2f}".format((2 * results[11] * results[12]) /
                                                                                    (results[11] + results[12])))

print("\n'Loss' del estadio anatomopatológico en el conjunto de prueba: {:.2f}\n""Sensibilidad del estadio "
      "anatomopatológico en el conjunto de prueba: {:.2f}\n""Precisión del estadio anatomopatológico en el conjunto de "
      "prueba: {:.2f}\n""Especifidad del estadio anatomopatológico en el conjunto de prueba: {:.2f} \n""Exactitud del "
      "estadio anatomopatológico en el conjunto de prueba: {:.2f} %\n""AUC-ROC del estadio anatomopatológico en el "
      "conjunto de prueba: {:.2f}".format(results[2], results[19], results[20], results[17]/(results[17]+results[16]),
                                          results[21] * 100, results[22]))
if results[19] > 0 or results[20] > 0:
    print("Valor-F del estadio anatomopatológico en el conjunto de prueba: {:.2f}".format((2 * results[19] * results[20]) /
                                                                                            (results[19] + results[20])))

print("\n'Loss' del parámetro 'T' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'T' en el conjunto de "
      "prueba: {:.2f}\n""Precisión del parámetro 'T' en el conjunto de prueba: {:.2f}\n""Especifidad del parámetro 'T' "
      "en el conjunto de prueba: {:.2f} \n""Exactitud del parámetro 'T' en el conjunto de prueba: {:.2f} %\n""AUC-ROC "
      "del parámetro 'T' en el conjunto de prueba: {:.2f}".format(results[3], results[27], results[28],
                                                                  results[25]/(results[25]+results[24]),
                                                                  results[29] * 100, results[30]))
if results[27] > 0 or results[28] > 0:
    print("Valor-F del parámetro 'T' en el conjunto de prueba: {:.2f}".format((2 * results[27] * results[28]) /
                                                                                (results[27] + results[28])))

print("\n'Loss' del parámetro 'N' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'N' en el conjunto de "
      "prueba: {:.2f}\n""Precisión del parámetro 'N' en el conjunto de prueba: {:.2f}\n""Especifidad del parámetro 'N' "
      "en el conjunto de prueba: {:.2f} \n""Exactitud del parámetro 'N' en el conjunto de prueba: {:.2f} %\n""AUC-ROC "
      "del parámetro 'N' en el conjunto de prueba: {:.2f}".format(results[4], results[35], results[36],
                                                                  results[33]/(results[33]+results[32]),
                                                                  results[37] * 100, results[38]))
if results[35] > 0 or results[36] > 0:
    print("Valor-F del parámetro 'N' en el conjunto de prueba: {:.2f}".format((2 * results[35] * results[36]) /
                                                                                (results[35] + results[36])))

print("\n'Loss' del parámetro 'M' en el conjunto de prueba: {:.2f}\n""Sensibilidad del parámetro 'M' en el conjunto de "
      "prueba: {:.2f}\n""Precisión del parámetro 'M' en el conjunto de prueba: {:.2f}\n""Especifidad del parámetro 'M' "
      "en el conjunto de prueba: {:.2f} \n""Exactitud del parámetro 'M' en el conjunto de prueba: {:.2f} %\n""AUC-ROC "
      "del parámetro 'M' en el conjunto de prueba: {:.2f}".format(results[5], results[43], results[44],
                                                                  results[41]/(results[41]+results[40]),
                                                                  results[45] * 100, results[46]))
if results[43] > 0 or results[44] > 0:
    print("Valor-F del parámetro 'M' en el conjunto de prueba: {:.2f}".format((2 * results[43] * results[44]) /
                                                                                (results[43] + results[44])))

print("\n'Loss' del subtipo molecular en el conjunto de prueba: {:.2f}\n""Sensibilidad del subtipo molecular en el "
      "conjunto de prueba: {:.2f}\n""Precisión del subtipo molecular en el conjunto de prueba: {:.2f}\n""Especifidad del "
      "subtipo molecular en el conjunto de prueba: {:.2f}\n""Exactitud del subtipo molecular en el conjunto de prueba: "
      "{:.2f} %\n""AUC-ROC del subtipo molecular en el conjunto de prueba: {:.2f}".format(results[6], results[51],
                                                                                          results[52],
                                                                                          results[49]/(results[49]+results[48]),
                                                                                          results[53] * 100,
                                                                                          results[54]))
if results[51] > 0 or results[52] > 0:
    print("Valor-F del subtipo molecular en el conjunto de prueba: {:.2f}".format((2 * results[51] * results[52]) /
                                                                                    (results[51] + results[52])))

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
conjunto de datos de test que se definió anteriormente al repartir los datos.
Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
# Tipo histológico
y_true_tumor_type = []
for label_test_tumor_type in test_labels_tumor_type:
    y_true_tumor_type.append(np.argmax(label_test_tumor_type))

y_true_tumor_type = np.array(y_true_tumor_type)
y_pred_tumor_type = np.argmax(model.predict(test_tabular_data)[0], axis = 1)

matrix_tumor_type = confusion_matrix(y_true_tumor_type, y_pred_tumor_type, labels = [0, 1, 2, 3, 4, 5, 6])
matrix_tumor_type_classes = ['IDC', 'ILC', 'Medullary', 'Metaplastic', 'Mixed (NOS)', 'Mucinous', 'Other']

# Estadio anatomopatológico
y_true_STAGE = []
for label_test_STAGE in test_labels_STAGE:
    y_true_STAGE.append(np.argmax(label_test_STAGE))

y_true_STAGE = np.array(y_true_STAGE)
y_pred_STAGE = np.argmax(model.predict(test_tabular_data)[1], axis = 1)

matrix_STAGE = confusion_matrix(y_true_STAGE, y_pred_STAGE, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Calcula (pero no dibuja) la matriz de confusión
matrix_STAGE_classes = ['Stage IB', 'Stage II', 'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IIIA', 'Stage IIIB',
                        'Stage IIIC', 'Stage IV', 'STAGE X']

# pT
y_true_pT = []
for label_test_pT in test_labels_pT:
    y_true_pT.append(np.argmax(label_test_pT))

y_true_pT = np.array(y_true_pT)
y_pred_pT = np.argmax(model.predict(test_tabular_data)[2], axis = 1)

matrix_pT = confusion_matrix(y_true_pT, y_pred_pT, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pT_classes = ['T1', 'T1A', 'T1B', 'T1C', 'T2', 'T2B', 'T3', 'T4', 'T4B', 'T4D']

# pN
y_true_pN = []
for label_test_pN in test_labels_pN:
    y_true_pN.append(np.argmax(label_test_pN))

y_true_pN = np.array(y_true_pN)
y_pred_pN = np.argmax(model.predict(test_tabular_data)[3], axis = 1)

matrix_pN = confusion_matrix(y_true_pN, y_pred_pN, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Calcula (pero no dibuja) la matriz de confusión
matrix_pN_classes = ['N1', 'N1A', 'N1B', 'N1C', 'N1MI', 'N2', 'N2A', 'N3', 'N3A', 'N3B', 'N3C']

# pM
y_true_pM = []
for label_test_pM in test_labels_pM:
    y_true_pM.append(np.argmax(label_test_pM))

y_true_pM = np.array(y_true_pM)
y_pred_pM = np.argmax(model.predict(test_tabular_data)[4], axis = 1)

matrix_pM = confusion_matrix(y_true_pM, y_pred_pM, labels = [0, 1, 2])
matrix_pM_classes = ['M0', 'M1', 'MX']

# IHQ
y_true_IHQ = []
for label_test_IHQ in test_labels_IHQ:
    y_true_IHQ.append(np.argmax(label_test_IHQ))

y_true_IHQ = np.array(y_true_IHQ)
y_pred_IHQ = np.argmax(model.predict(test_tabular_data)[5], axis = 1)

matrix_IHQ = confusion_matrix(y_true_IHQ, y_pred_IHQ, labels = [0, 1, 2, 3, 4]) # Calcula (pero no dibuja) la matriz de confusión
matrix_IHQ_classes = ['Basal', 'Her2', 'Luminal A', 'Luminal B', 'Normal']

""" Función para mostrar por pantalla la matriz de confusión multiclase con todas las clases de subtipos moleculares """
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión', cmap = plt.cm.Blues):
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

plot_confusion_matrix(matrix_tumor_type, classes = matrix_tumor_type_classes, title = 'Matriz de confusión del tipo '
                                                                                      'histológico')
plt.show()

plot_confusion_matrix(matrix_STAGE, classes = matrix_STAGE_classes, title = 'Matriz de confusión del estadio '
                                                                            'anatomopatológico')
plt.show()

plot_confusion_matrix(matrix_pT, classes = matrix_pT_classes, title = 'Matriz de confusión del parámetro "T"')
plt.show()

plot_confusion_matrix(matrix_pN, classes = matrix_pN_classes, title = 'Matriz de confusión del parámetro "N"')
plt.show()

plot_confusion_matrix(matrix_pM, classes = matrix_pM_classes, title = 'Matriz de confusión del parámetro "M"')
plt.show()

plot_confusion_matrix(matrix_IHQ, classes = matrix_IHQ_classes, title = 'Matriz de confusión del subtipo molecular')
plt.show()

#np.save('test_data', test_tabular_data)
#np.save('test_labels_tumor_type', test_labels_tumor_type)
#np.save('test_labels_STAGE', test_labels_STAGE)
#np.save('test_labels_pT', test_labels_pT)
#np.save('test_labels_pN', test_labels_pN)
#np.save('test_labels_pM', test_labels_pM)
#np.save('test_labels_IHQ', test_labels_IHQ)