import shelve # datos persistentes
import pandas as pd
import numpy as np
import imblearn
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
#filename = 'C:\\Users\\valde\Desktop\Datos_repositorio\cbioportal\data/brca_tcga_pan_can_atlas_2018.out'

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

df_list = [df_age, df_neoadjuvant, df_path_m_stage, df_path_n_stage, df_path_t_stage, df_stage, df_subtype,
           df_tumor_type, df_prior_diagnosis, df_snv, df_cnv]

""" Fusionar todos los dataframes (los cuales se han recopilado en una lista) por la columna 'ID' para que ningún valor
esté descuadrado en la fila que no le corresponda. """
df_all_merge = reduce(lambda left,right: pd.merge(left,right,on=['ID'], how='left'), df_list)

""" Ahora se va a encontrar cuales son los ID de los genes que nos interesa. Para ello se crean dos variables para
crear una lista de claves y otra de los valores del diccionario de genes. Se extrae el índice de los genes en la lista 
de valores y posteriormente se usan esos índices para buscar con qué claves (ID) se corresponden en la lista de claves. 
Se almacenan todos los IDs de los genes en una lista. """
snv_list = ['PIK3CA', 'TP53', 'PTEN', 'ERBB2', 'AKT1', 'MTOR', 'EGFR']
id_snv_list = []

cnv_list = ['MYC' , 'CCND1', 'CDKN1B', 'FGF19', 'ERBB2', 'FGF3', 'BRCA2' , 'BRCA1', 'KDR', 'CHEK1', 'FANCA']
id_cnv_list = []

key_list = list(dict_genes.keys())
val_list = list(dict_genes.values())

for gen_snv in snv_list:
    position = val_list.index(gen_snv) # Número
    id_gen_snv = (key_list[position]) # Número
    id_snv_list.append(id_gen_snv) # Se añaden todos los IDs en la lista vacía

for gen_cnv in cnv_list:
    position = val_list.index(gen_cnv) # Número
    id_gen_cnv = (key_list[position]) # Número
    id_cnv_list.append(id_gen_cnv) # Se añaden todos los IDs en la lista vacía

""" Se hace un bucle sobre la columna de mutaciones del dataframe. Así, se busca en cada mutación de cada fila para ver
en que filas encuentra el ID del gen que se quiere predecir. Se almacenan en una lista de listas los índices de las 
filas donde se encuentran esos IDs de esos genes, de forma que se tiene una lista para cada gen. """
# SNV:
list_gen_snv = [[] for ID in range(7)]

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

""" Se recopila los índices de las distintas filas donde aparecen las mutaciones 'CNV' de los genes seleccionados (tanto 
de amplificación como deleción), y se añaden a la lista de listas correspondiente (la de amplificación o la de deleción). """
list_gen_cnv_amp = [[] for ID in range(11)]
list_gen_cnv_del = [[] for ID in range(11)]

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

    else:
        for index_row, row in enumerate (df_all_merge['CNV']): # Para cada fila dentro de la columna 'SNV'...
            for mutation in row: # Para cada mutación dentro de cada fila...
                if mutation[1] == id_cnv and mutation[2] > 0:
                    list_gen_cnv_amp[index].append(index_row) # Se almacena el índice de la fila en la lista de listas
                elif mutation[1] == id_cnv and mutation[2] < 0:
                    list_gen_cnv_del[index].append(index_row) # Se almacena el índice de la fila en la lista de listas

""" Una vez se tienen almacenados los índices de las filas donde se producen esas mutaciones, hay que crear distintas
columnas que nos dirán si para el paciente en cuestión, éste tiene o no mutación en un determinado gen de los 
seleccionados. """
# SNV:
df_all_merge.rename(columns={'SNV': 'SNV_PIK3CA'}, inplace= True)
df_all_merge['SNV_PIK3CA'], df_all_merge['SNV_TP53'], df_all_merge['SNV_PTEN'], df_all_merge['SNV_ERBB2'], \
df_all_merge['SNV_AKT1'], df_all_merge['SNV_MTOR'], df_all_merge['SNV_EGFR'] = [0, 0, 0, 0, 0, 0, 0]

# CNV:
df_all_merge.rename(columns={'CNV': 'CNV_MYC_AMP'}, inplace= True)
df_all_merge['CNV_MYC_AMP'], df_all_merge['CNV_MYC_NORMAL'], df_all_merge['CNV_MYC_DEL'], \
df_all_merge['CNV_CCND1_AMP'],df_all_merge['CNV_CCND1_NORMAL'], df_all_merge['CNV_CCND1_DEL'], \
df_all_merge['CNV_CDKN1B_AMP'], df_all_merge['CNV_CDKN1B_NORMAL'], df_all_merge['CNV_CDKN1B_DEL'], \
df_all_merge['CNV_FGF19_AMP'], df_all_merge['CNV_FGF19_NORMAL'], df_all_merge['CNV_FGF19_DEL'], \
df_all_merge['CNV_ERBB2_AMP'], df_all_merge['CNV_ERBB2_NORMAL'], df_all_merge['CNV_ERBB2_DEL'], \
df_all_merge['CNV_FGF3_AMP'], df_all_merge['CNV_FGF3_NORMAL'], df_all_merge['CNV_FGF3_DEL'], \
df_all_merge['CNV_BRCA2_AMP'], df_all_merge['CNV_BRCA2_NORMAL'], df_all_merge['CNV_BRCA2_DEL'], \
df_all_merge['CNV_BRCA1_AMP'], df_all_merge['CNV_BRCA1_NORMAL'], df_all_merge['CNV_BRCA1_DEL'], \
df_all_merge['CNV_KDR_AMP'], df_all_merge['CNV_KDR_NORMAL'], df_all_merge['CNV_KDR_DEL'], \
df_all_merge['CNV_CHEK1_AMP'], df_all_merge['CNV_CHEK1_NORMAL'], df_all_merge['CNV_CHEK1_DEL'], \
df_all_merge['CNV_FANCA_AMP'], df_all_merge['CNV_FANCA_NORMAL'], df_all_merge['CNV_FANCA_DEL'] = [0] * 33

""" Una vez han sido creadas las columnas, se añade un '1' en aquellas filas donde el paciente tiene mutación sobre el
gen seleccionado. Se utiliza para ello los índices recogidos anteriormente en las respectivas listas de listas. """
# SNV:
for index in list_gen_snv[0]: # Para cada índice dentro de la sublista cero (que es la del gen PIK3CA)...
    df_all_merge.loc[index, 'SNV_PIK3CA'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_snv[1]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'SNV_TP53'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_snv[2]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'SNV_PTEN'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_snv[3]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'SNV_ERBB2'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_snv[4]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'SNV_AKT1'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_snv[5]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'SNV_MTOR'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_snv[6]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'SNV_EGFR'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

# CNV:
for index in list_gen_cnv_amp[0]: # Para cada índice dentro de la sublista cero (que es la del gen MYC)...
    df_all_merge.loc[index, 'CNV_MYC_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[1]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_CCND1_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[2]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_CDKN1B_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[3]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_FGF19_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[4]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_ERBB2_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[5]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_FGF3_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[6]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_BRCA2_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[7]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_BRCA1_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[8]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_KDR_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[9]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_CHEK1_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_amp[10]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_FANCA_AMP'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[0]: # Para cada índice dentro de la sublista cero (que es la del gen MYC)...
    df_all_merge.loc[index, 'CNV_MYC_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[1]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_CCND1_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[2]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_CDKN1B_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[3]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_FGF19_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[4]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_ERBB2_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[5]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_FGF3_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[6]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_BRCA2_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[7]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_BRCA1_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[8]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_KDR_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[9]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_CHEK1_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

for index in list_gen_cnv_del[10]: # Para cada índice dentro de la sublista del gen...
    df_all_merge.loc[index, 'CNV_FANCA_DEL'] = 1 # Se escribe un '1' en la fila que indica el índice de la sublista

""" Falta por rellenar la columna normal de todas las mutaciones CNV. Para ello se colocará un '1' en aquellas filas 
donde no haya mutación CNV de amplificación o deleción para un determinado gen. """
for index in range(1084):
    if index not in list_gen_cnv_amp[0] and index not in list_gen_cnv_del[0]:
        df_all_merge.loc[index, 'CNV_MYC_NORMAL'] = 1

    if index not in list_gen_cnv_amp[1] and index not in list_gen_cnv_del[1]:
        df_all_merge.loc[index, 'CNV_CCND1_NORMAL'] = 1

    if index not in list_gen_cnv_amp[2] and index not in list_gen_cnv_del[2]:
        df_all_merge.loc[index, 'CNV_CDKN1B_NORMAL'] = 1

    if index not in list_gen_cnv_amp[3] and index not in list_gen_cnv_del[3]:
        df_all_merge.loc[index, 'CNV_FGF19_NORMAL'] = 1

    if index not in list_gen_cnv_amp[4] and index not in list_gen_cnv_del[4]:
        df_all_merge.loc[index, 'CNV_ERBB2_NORMAL'] = 1

    if index not in list_gen_cnv_amp[5] and index not in list_gen_cnv_del[5]:
        df_all_merge.loc[index, 'CNV_FGF3_NORMAL'] = 1

    if index not in list_gen_cnv_amp[6] and index not in list_gen_cnv_del[6]:
        df_all_merge.loc[index, 'CNV_BRCA2_NORMAL'] = 1

    if index not in list_gen_cnv_amp[7] and index not in list_gen_cnv_del[7]:
        df_all_merge.loc[index, 'CNV_BRCA1_NORMAL'] = 1

    if index not in list_gen_cnv_amp[8] and index not in list_gen_cnv_del[8]:
        df_all_merge.loc[index, 'CNV_KDR_NORMAL'] = 1

    if index not in list_gen_cnv_amp[9] and index not in list_gen_cnv_del[9]:
        df_all_merge.loc[index, 'CNV_CHEK1_NORMAL'] = 1

    if index not in list_gen_cnv_amp[10] and index not in list_gen_cnv_del[10]:
        df_all_merge.loc[index, 'CNV_FANCA_NORMAL'] = 1

""" En este caso, se eliminan los pacientes con categoria 'N0 o 'NX', y tambien aquellos pacientes a los que no se les 
puede determinar si tienen o no metastasis, porque sino el problema ya no seria de clasificacion binaria. """
df_all_merge = df_all_merge[(df_all_merge["path_n_stage"]!='N0') & (df_all_merge["path_n_stage"]!='NX') &
                            (df_all_merge["path_n_stage"]!='N0 (I-)') & (df_all_merge["path_n_stage"]!='N0 (I+)') &
                            (df_all_merge["path_n_stage"]!='N0 (MOL+)') & (df_all_merge["path_m_stage"]!='MX')]

""" Al realizar un análisis de los datos de entrada se ha visto un único valor incorrecto en la columna
'cancer_type_detailed'. Por ello se sustituye dicho valor por 'Breast Invasive Carcinoma (NOS)'. También se ha apreciado
un único valor en 'tumor_type', por lo que también se realiza un cambio de valor en dicho valor atípico. Además, se 
convierten las columnas categóricas binarias a valores de '0' y '1', para no aumentar el número de columnas: """
df_all_merge.loc[df_all_merge.tumor_type == "Infiltrating Carcinoma (NOS)", "tumor_type"] = "Mixed Histology (NOS)"
df_all_merge.loc[df_all_merge.tumor_type == "Breast Invasive Carcinoma", "tumor_type"] = "Infiltrating Ductal Carcinoma"
df_all_merge.loc[df_all_merge.neoadjuvant == "No", "neoadjuvant"] = 0; df_all_merge.loc[df_all_merge.neoadjuvant == "Yes", "neoadjuvant"] = 1
df_all_merge.loc[df_all_merge.prior_diagnosis == "No", "prior_diagnosis"] = 0; df_all_merge.loc[df_all_merge.prior_diagnosis == "Yes", "prior_diagnosis"] = 1
df_all_merge.loc[df_all_merge.path_m_stage == "CM0 (I+)", "path_m_stage"] = 'M0'
df_all_merge.loc[df_all_merge.path_m_stage == "M0", "path_m_stage"] = 0; df_all_merge.loc[df_all_merge.path_m_stage == "M1", "path_m_stage"] = 1

""" Ahora, antes de transformar las variables categóricas en numéricas, se eliminan las filas donde haya datos nulos
para no ir arrastrándolos a lo largo del programa: """
df_all_merge.dropna(inplace=True) # Mantiene el DataFrame con las entradas válidas en la misma variable.

""" Una vez la tabla tiene las columnas deseadas se procede a codificar las columnas categóricas del dataframe a valores
numéricos mediante la técnica del 'One Hot Encoding'. Más adelante se escalarán las columnas numéricas continuas, pero
ahora se realiza esta técnica antes de hacer la repartición de subconjuntos para que no haya problemas con las columnas. """
#@ get_dummies: Aplica técnica de 'One Hot Encoding', creando columnas binarias para las columnas seleccionadas
df_all_merge = pd.get_dummies(df_all_merge, columns=["path_n_stage", "path_t_stage", "stage", "subtype","tumor_type"])

""" Se dividen los datos tabulares y las imágenes con cáncer en conjuntos de entrenamiento y test con @train_test_split.
Con @random_state se consigue que en cada ejecución la repartición sea la misma, a pesar de estar barajada: """
train_tabular_data, test_tabular_data = train_test_split(df_all_merge, test_size = 0.20,
                                                         stratify = df_all_merge['path_m_stage'])

train_tabular_data, valid_tabular_data = train_test_split(train_tabular_data, test_size = 0.20,
                                                          stratify = train_tabular_data['path_m_stage'])

""" Ya e puede eliminar de los dos subconjuntos la columna 'ID' que no es útil para la red MLP: """
#@inplace = True para que devuelva el resultado en la misma variable
train_tabular_data.drop(['ID'], axis=1, inplace= True)
valid_tabular_data.drop(['ID'], axis=1, inplace= True)
test_tabular_data.drop(['ID'], axis=1, inplace= True)

""" Se extrae la columna 'path_m_stage' del dataframe de ambos subconjuntos, puesto que ésta es la salida del modelo que se
va a entrenar."""
train_labels = train_tabular_data.pop('path_m_stage')
valid_labels = valid_tabular_data.pop('path_m_stage')
test_labels = test_tabular_data.pop('path_m_stage')

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
train_labels = np.asarray(train_labels).astype('float32')

valid_tabular_data = np.asarray(valid_tabular_data).astype('float32')
valid_labels = np.asarray(valid_labels).astype('float32')

test_tabular_data = np.asarray(test_tabular_data).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

""" -------------------------------------------------------------------------------------------------------------------
---------------------------------- SECCIÓN MODELO DE RED NEURONAL (MLP) -----------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
model = keras.Sequential()
model.add(layers.Dense(train_tabular_data.shape[1], activation='relu', input_shape=(train_tabular_data.shape[1],)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(46, activation = "relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = "sigmoid"))
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
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = metrics)

""" Se implementa un callback: para guardar el mejor modelo que tenga la menor 'loss' en la validación. """
checkpoint_path = '../../training_codes/data/data_model_distant_metastasis_prediction.h5'
mcp_save = ModelCheckpoint(filepath= checkpoint_path, save_best_only = True, monitor= 'val_loss', mode= 'min')

smoter = imblearn.over_sampling.SMOTE(sampling_strategy = 'minority')
train_tabular_data, train_labels = smoter.fit_resample(train_tabular_data, train_labels)

""" Esto se hace para que al hacer el entrenamiento, los pesos de las distintas salidas se balaceen, ya que el conjunto
de datos que se tratan en este problema es muy imbalanceado. """
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels),
                                                  y = train_labels)
class_weight_dict = dict(enumerate(class_weights))

""" Una vez definido y compilado el modelo, es hora de entrenarlo. """
neural_network = model.fit(x = train_tabular_data,  # Datos de entrada.
                           y = train_labels,  # Datos objetivos.
                           epochs = 150,
                           verbose = 1,
                           batch_size = 32,
                           class_weight = class_weight_dict,
                           #callbacks= mcp_save,
                           validation_data = (valid_tabular_data, valid_labels))

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_tabular_data, test_labels, verbose = 0)
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
# @suppress = True: Muestra los números con representación de coma fija
# @predict: Genera predicciones para nuevas entradas
print("\nGenera predicciones para 10 muestras")
print("Clase de las salidas: ", test_labels[:10])
np.set_printoptions(precision=3, suppress=True)
print("Predicciones:\n", np.round(model.predict(test_tabular_data[:10])))

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
y_true = test_labels # Etiquetas verdaderas de 'test'
y_pred = np.round(model.predict(test_tabular_data)) # Predicción de etiquetas de 'test'

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

y_pred_prob = model.predict(test_tabular_data).ravel()
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
auc_roc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label = 'No Skill')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('AUC-ROC curve')
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
plt.title('AUC-PR curve')
plt.legend(loc = 'best')
plt.show()

#np.save('test_data', test_tabular_data)
#np.save('test_labels', test_labels)