import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from tensorflow.keras.models import load_model
import seaborn as sns

""" Se carga el Excel de INiBICA y se recopilan las salidas de los genes CNV-A """
data_inibica = pd.read_excel('/home/avalderas/img_slides/excel_genesOCA&inibica_patients/inference_inibica.xlsx',
                              engine='openpyxl')

test_labels_cnv_a = data_inibica.iloc[:, 167::3]

""" Ademas se recopilan los nombres de las columnas para usarlos posteriormente """
test_columns_cnv_a = test_labels_cnv_a.columns.values
classes_cnv_a = test_columns_cnv_a.tolist()

""" Parametros de las teselas """
ancho = 210
alto = 210
canales = 3

""" Se abre WSI especificada """
path_wsi = '/home/avalderas/img_slides/wsi/397W_HE_40x.tiff'
wsi = openslide.OpenSlide(path_wsi)

"""" Se hallan las dimensiones (anchura, altura) del nivel de resolución '0' (máxima resolución) de la WSI """
dim = wsi.dimensions

""" Se averigua cual es el mejor nivel de resolución de la WSI para mostrar el factor de reduccion deseado """
best_level = wsi.get_best_level_for_downsample(10) # factor de reducción deseado

""" Se averigua cual es el factor de reducción de dicho nivel para usarlo posteriormente al multiplicar las dimensiones
en la función @read_region"""
scale = int(wsi.level_downsamples[best_level])
score_tiles = []

all_tiles = np.zeros((int(dim[0]/(ancho * scale)), int(dim[1] / (alto * scale)))) # (120,81) = 9720 teselas en nivel 1
white_pixel = []

""" Se itera sobre todas las teselas de tamaño 210x210 de la WSI en el nivel adecuado al factor de reduccion '10x'. 
Se comprueban las teselas vacias (en blanco) convirtiendo la tesela en blanco (0) y negro (1) y comparando la cantidad 
de valores que no son cero con respecto las dimensiones de las teselas (210x210). Si dicha comparacion supera el umbral 
de 0.1, la tesela no esta vacia y se añade a la lista de teselas a las que realizarle la inferencia. """
for alto_slide in range(int(dim[1]/(alto*scale))):
    for ancho_slide in range(int(dim[0] / (ancho * scale))):
        sub_img = wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                           (ancho, alto))
        sub_img = sub_img.convert('1') # Blanco y negro
        score = 1 - (np.count_nonzero(sub_img) / (ancho * alto))
        all_tiles[ancho_slide][alto_slide] = 1 - (np.count_nonzero(sub_img) / (ancho * alto)) # Puntuacion (0 - 1)
        white_pixel.append(all_tiles) # Lista de 9720 teselas. Cada una con forma (120, 81)

        if score > 0.1: # Teselas que no son blancas y contienen algo en la imagen (puntuacion > 0.1)
            sub_img = np.array(wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                               (ancho, alto)))
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGBA2RGB)
            #plt.title('Imagen RGBA de la región especificada')
            #plt.imshow(sub_img)
            #plt.show()
            #cv2.imshow('tesela', sub_img)
            #cv2.waitKey(0)
            score_tiles.append(sub_img)

score_tiles = np.array(score_tiles) # 4377, 210, 210, 3

# Generate a heatmap
#grid = white_pixel[0]
#sns.set(style="white")
#plt.subplots(figsize=(grid.shape[0]/5, grid.shape[1]/5))

#mask = np.zeros_like(grid)
#mask[np.where(grid < 0.1)] = True #Mask blank tiles

#sns.heatmap(grid.T, square=True, linewidths=.5, mask=mask.T, cbar=False, vmin=0, vmax=1, cmap="Reds")
#plt.show()

#grid = None
#white_pixel = None

""" Se carga el modelo """
model = load_model('/home/avalderas/img_slides/mutations/image/inference/test_data&models/model_image_mutations.h5')

cnv_a = []
cnv_a_scores = []

""" Se realiza la prediccion de los 43 genes CNV-A para cada una de las teselas de la imagen y se añaden dichas
predicciones a una lista. """
for tile in score_tiles:
    tile = np.expand_dims(tile, axis = 0)
    cnv_a.append(model.predict(tile)[1])
    cnv_a_scores.append(np.sum(model.predict(tile)[1], axis = 1))

""" Se realiza la suma para cada una de las columnas de la lista de predicciones. Como resultado, se obtiene una lista
de 43 columnas y 1 sola fila, ya que se han sumado las predicciones de todas las teselas para cada gen. """
#cnv_a = np.concatenate(cnv_a)
#cnv_a_sum_columns = cnv_a.sum(axis = 0)

""" Se ordenan los indices de la lista resultante ordenador de mayor a menor valor, mostrando solo el resultado con
mayor valor, que sera el de la mutacion CNV-A mas probable """
#mpm_cnv_a = np.argsort(cnv_a_sum_columns)[::-1]

#for (i, j) in enumerate(mpm_cnv_a):
    #label_cnv_a = "\nLa mutación CNV-A más probable es del gen {}: {:.2f}%".format(classes_cnv_a[j].split('_')[1],
                                                                                   #proba[j] * 100)
    #print(label_cnv_a)

# Generate a heatmap
grid = white_pixel[0]
sns.set(style="white")
plt.subplots(figsize=(grid.shape[0]/5, grid.shape[1]/5))

mask = np.zeros_like(grid)
mask[np.where(grid < 0.1)] = True # Enmascara las teselas blancas

sns.heatmap(np.array(cnv_a_scores).T, square = True, linewidths = .5, mask = mask.T, cbar = True, vmin = 0, vmax = 1,
            cmap = "Reds")
plt.show()

results = model.evaluate(test_image_data, [test_labels_snv, test_labels_cnv_a, test_labels_cnv_normal,
                                           test_labels_cnv_d], verbose = 0)
