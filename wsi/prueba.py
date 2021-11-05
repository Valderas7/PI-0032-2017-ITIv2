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

""" Se carga el modelo de la red neuronal """
model = load_model('/home/avalderas/img_slides/mutations/image/inference/test_data&models/model_image_mutations.h5')

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

""" Se crea un 'array' con forma (81, 120), que son el número de filas y el número de columnas, respectivamente, en el 
que se divide la WSI al dividirla en teselas de 210x210 para recopilar asi las puntuaciones de cada tesela """
tiles_scores_array = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Se crea también una lista para recopilar las puntuaciones de cada tesela """
tiles_scores_list = []

""" Listas para recopilar las predicciones y las puntuaciones de las mutaciones CNV-A """
cnv_a = []
cnv_a_scores = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Se itera sobre todas las teselas de tamaño 210x210 de la WSI en el nivel adecuado al factor de reduccion '10x' """
#@ancho_slide itera de (0-119) [columnas] y @alto_slide de (0-80) [filas]
for alto_slide in range(int(dim[1]/(alto*scale))):
    for ancho_slide in range(int(dim[0] / (ancho * scale))):
        sub_img = wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                           (ancho, alto))

        """ Se comprueban las teselas vacías (en blanco) convirtiendo la tesela en blanco (0) y negro (1) y comparando
        la cantidad de píxeles que no son 0 (blanco) con respecto las dimensiones de las teselas (210x210). Si dicha
        comparación supera el umbral de 0.1, la tesela no está vacía y se añade a la lista de teselas a las que
        realizarle la inferencia. """
        sub_img = sub_img.convert('1') # Blanco y negro
        score = 1 - (np.count_nonzero(sub_img) / (ancho * alto))

        """ Puntuación para cada tesela: 'tiles_scores_array' es una matriz de (120, 81) donde se recogen todas las 
        puntuaciones para cada tesela colocadas correctamente según su fila y columna. De esta forma, 
        'tiles_scores_array' imprime el array de 2D de las puntuaciones de todas las teselas; y 
        'tiles_scores_array[alto_slide][ancho_slide]' imprime la puntuación de una tesela en la fila [alto_slide] y en 
        la columna [ancho_slide] """
        tiles_scores_array[alto_slide][ancho_slide] = score

        if tiles_scores_array[alto_slide][ancho_slide] > 0.1:
            sub_img = np.array(wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                               (ancho, alto)))
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGBA2RGB)
            #tile = np.expand_dims(sub_img, axis = 0)
            #cnv_a.append(model.predict(tile)[1])
            #cnv_a_scores[alto_slide][ancho_slide] = (np.sum(model.predict(tile)[1], axis = 1))

""" La lista 'tiles_scores_list' es una lista 3D donde se almacenan las puntuaciones de las teselas. En esta ocasion, la 
lista tiene una forma de (1, 81, 120), es decir, hay 1 matriz con las puntuaciones de las teselas en forma de lista, 
siendo ésta 81 filas y 120 columnas. """
tiles_scores_list.append(tiles_scores_array)

""" Se dibuja un mapa de calor según las predicciones (los datos de entrada deben estar en 2D) """
grid = tiles_scores_list[0] # (81, 120)
slide = np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1]))
slide = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)

""" El tamaño de las figuras en Python se establece en pulgadas (inches). La fórmula para convertir de píxeles a 
pulgadas es (píxeles / dpi = pulgadas). La cantidad de píxeles permanece invariable porque depende de la imagen de menor
resolucion de la WSI. Por lo tanto, para convertir a pulgadas, hay que variar el DPI (puntos por pulgada) y las propias
pulgadas. """
pixeles_x = slide.shape[1]
pixeles_y = slide.shape[0]
dpi = 96

sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se crea una máscara para las puntuaciones menores de 0.1, de forma que no es pasan datos en aquellas celdas donde no
se alcanza dicha puntuación """
mask = np.zeros_like(grid)
mask[np.where(tiles_scores_list[0] < 0.1)] = True

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2)

""" Se adapta la imagen del WSI a las dimensiones del mapa de calor """
heatmap.imshow(np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1])),
               aspect = heatmap.get_aspect(), extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1)
#plt.figure()
plt.show()
quit()
""" Hay que aplicar ese mapa de calor sobre la imagen """
image_wsi_min_resolution = np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1]))
image_wsi_min_resolution = cv2.cvtColor(image_wsi_min_resolution, cv2.COLOR_RGBA2RGB)
heatmap_array = cv2.mat(heatmap)
plt.figure()
#plt.imshow(heatmap_array)
plt.show()

merge = cv2.addWeighted(src1 = image_wsi_min_resolution, alpha = 0.7, src2 = heatmap_array, beta = 0.3, gamma = 0.0)
plt.figure()
plt.imshow(merge)
plt.show()
cv2.imshow('result', merge)
cv2.waitKey(0)
#grid = None
#white_pixel = None


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


results = model.evaluate(test_image_data, [test_labels_snv, test_labels_cnv_a, test_labels_cnv_normal,
                                           test_labels_cnv_d], verbose = 0)
