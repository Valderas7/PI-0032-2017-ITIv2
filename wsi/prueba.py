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

""" Se averigua cual es el mejor nivel para mostrar el factor de reduccion deseado"""
best_level = wsi.get_best_level_for_downsample(10) # factor de reducción deseado

'''@dimensions: Método que devuelve la (anchura, altura) del nivel de resolución '0' (máxima resolución) de la imagen'''
dim = wsi.dimensions

""" Se averigua cual es el factor de reduccion del nivel que mejor representa las imagenes entrenadas. Se multiplica 
este factor de reduccion por las dimensiones del nivel de resolucion maximo, que es el nivel de referencia para la 
funcion @read_region """
scale = int(wsi.level_downsamples[best_level])
tiles = []

white_tiles = np.zeros((int(dim[0]/(ancho * scale)), int(dim[1] / (alto * scale))))
white_pixel = []

""" Se itera sobre todas las teselas de tamaño 210x210 de la WSI en el nivel adecuado al factor de reduccion '10x'. 
Se comprueban las teselas vacias (en blanco) convirtiendo la tesela en blanco (0) y negro (1) y comparando la cantidad 
de valores que no son cero con respecto las dimensioens de las teselas (210x210). Si dicha comparacion supera el umbral 
de 0.1, la tesela no esta vacia y se añade a la lista de teselas a las que realizarle la inferencia. """
for alto_slide in range(int(dim[1]/(alto*scale))):
    for ancho_slide in range(int(dim[0] / (ancho * scale))):
        sub_img = wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                           (ancho, alto))
        sub_img = sub_img.convert('1') # Blanco y negro
        score = 1 - (np.count_nonzero(sub_img) / (ancho * alto))
        white_tiles[ancho_slide][alto_slide] = 1 - (np.count_nonzero(sub_img) / (ancho * alto))
        white_pixel.append(white_tiles)

        if score > 0.1:
            sub_img = np.array(wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                               (ancho, alto)))
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGBA2RGB)
            #plt.title('Imagen RGBA de la región especificada')
            #plt.imshow(sub_img)
            #plt.show()
            #cv2.imshow('tesela', sub_img)
            #cv2.waitKey(0)
            tiles.append(sub_img)

tiles = np.array(tiles)

# Generate a heatmap
grid = white_pixel[0]
sns.set(style="white")
plt.subplots(figsize=(grid.shape[0]/5, grid.shape[1]/5))

mask = np.zeros_like(grid)
mask[np.where(grid < 0.1)] = True #Mask blank tiles

sns.heatmap(grid.T, square=True, linewidths=.5, mask=mask.T, cbar=False, vmin=0, vmax=1, cmap="Reds")
plt.show()
print('Not-blank tiles:', np.count_nonzero(grid), 'on', grid.size, 'total tiles')
grid = None
white_pixel = None

""" Se carga el modelo """
model = load_model('/home/avalderas/img_slides/mutations/image/inference/test_data&models/model_image_mutations.h5')

cnv_a = []

""" Se realiza la prediccion de los 43 genes CNV-A para cada una de las teselas de la imagen y se añaden dichas
predicciones a una lista. """
for tile in tiles:
    tile = np.expand_dims(tile, axis = 0)
    cnv_a.append(model.predict(tile)[1])

""" Se realiza la suma para cada una de las columnas de la lista de predicciones. Como resultado, se obtiene una lista
de 43 columnas y 1 sola fila, ya que se han sumado las predicciones de todas las teselas para cada gen. """
cnv_a = np.concatenate(cnv_a)
cnv_a_sum_columns = cnv_a.sum(axis = 0)

""" Se ordenan los indices de la lista resultante ordenador de mayor a menor valor, mostrando solo el resultado con
mayor valor, que sera el de la mutacion CNV-A mas probable """
mpm_cnv_a = np.argsort(cnv_a_sum_columns)[::-1]

for (i, j) in enumerate(mpm_cnv_a):
    label_cnv_a = "\nLa mutación CNV-A más probable es del gen {}: {:.2f}%".format(classes_cnv_a[j].split('_')[1],
                                                                                   proba[j] * 100)
    print(label_cnv_a)

results = model.evaluate(test_image_data, [test_labels_snv, test_labels_cnv_a, test_labels_cnv_normal,
                                           test_labels_cnv_d], verbose = 0)
