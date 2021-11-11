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
path_wsi = '/media/proyectobdpath/PI0032WEB/P001-HE-014-2.mrxs'
wsi = openslide.OpenSlide(path_wsi)

"""" Se hallan las dimensiones (anchura, altura) del nivel de resolución '0' (máxima resolución) de la WSI """
dim = wsi.dimensions

""" Se averigua cual es el mejor nivel de resolución de la WSI para mostrar el factor de reduccion deseado """
best_level = wsi.get_best_level_for_downsample(10) # factor de reducción deseado

""" Se averigua cual es el factor de reducción de dicho nivel para usarlo posteriormente al multiplicar las dimensiones
en la función @read_region """
scale = int(wsi.level_downsamples[best_level])
score_tiles = []

""" Para saber en que nivel es mejor mostrar posteriormente el mapa de calor, hace falta saber cuántos niveles hay en la
WSI """
levels = wsi.level_count

""" Una vez se sabe el número de niveles de resolución, hay que encontrar un nivel lo suficientemente grande que tenga 
unas dimensiones lo suficientemente grandes para crear un mapa de calor con gran resolución, y que al mismo tiempo sea
lo suficientemente pequeño para que el software pueda realizar la computación de dichos niveles """
dimensions_map = 0
level_map = 0

for level in range(levels):
    if wsi.level_dimensions[level][1] <= 6666:
        dimensions_map = wsi.level_dimensions[level]
        level_map = level
        break

""" Se crea un 'array' con forma (alto, ancho), que son el número de filas y el número de columnas, respectivamente, en 
el que se divide la WSI al dividirla en teselas de 210x210 en el nivel de resolucion máximo, para recopilar asi las 
puntuaciones de cada tesela """
tiles_scores_array = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Se crea también una lista para recopilar las puntuaciones de cada tesela """
tiles_scores_list = []

""" Se crea una lista para recopilar las predicciones de las mutaciones CNV-A y un 'array' en 3D para recopilar las 
puntuaciones de las distintas mutaciones de los genes. """
cnv_a = []
cnv_a_genes = 5 # MYC, CCND1, FGF19, ERBB2, FGF3. CDKN1B NO ESTA EN EL PANEL OCA.
cnv_a_scores = np.zeros((cnv_a_genes, int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Se itera sobre todas las teselas de tamaño 210x210 de la WSI en el nivel adecuado al factor de reduccion '10x' """
#@ancho_slide itera de (0- nºcolumnas) [columnas] y @alto_slide de (0- nºfilas) [filas]
for alto_slide in range(int(dim[1]/(alto*scale))):
    for ancho_slide in range(int(dim[0] / (ancho * scale))):
        sub_img = wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level, (ancho, alto))

        """ Se comprueban las teselas vacías (en blanco) convirtiendo la tesela en blanco (0) y negro (1) y comparando
        la cantidad de píxeles que no son 0 (blanco) con respecto las dimensiones de las teselas (210x210). Si dicha
        comparación supera el umbral de 0.1, la tesela no está vacía y se añade a la lista de teselas a las que
        realizarle la inferencia. """
        sub_img_black_white = sub_img.convert('1') # Blanco y negro
        score = 1 - (np.count_nonzero(sub_img_black_white) / (ancho * alto))

        """ Puntuación para cada tesela: 'tiles_scores_array' es una matriz de (120, 81) donde se recogen todas las 
        puntuaciones para cada tesela colocadas correctamente según su fila y columna. De esta forma, 
        'tiles_scores_array' imprime el array de 2D de las puntuaciones de todas las teselas; y 
        'tiles_scores_array[alto_slide][ancho_slide]' imprime la puntuación de una tesela en la fila [alto_slide] y en 
        la columna [ancho_slide] """
        tiles_scores_array[alto_slide][ancho_slide] = score

        if 0.2 < tiles_scores_array[alto_slide][ancho_slide] < 0.9:
            """ Primero se intenta hallar si hay una línea recta negra que dura todo el ancho de la tesela. Para ello se
            itera sobre todas las filas de los tres canales RGB de la tesela para saber si en algún momento la suma de 
            tres filas correspodientes en los tres canales de la tesela es cero, lo que indicaría que hay una fila 
            entera en la tesela de color negro, y por tanto, se hace que su puntuacion sea cero. """
            sub_img_array = cv2.cvtColor(np.array(sub_img), cv2.COLOR_RGBA2RGB)
            r = 1; g = 2; b = 3
            for index in range(210):
                r = np.sum(sub_img_array[index, :, 0])
                g = np.sum(sub_img_array[index, :, 1])
                b = np.sum(sub_img_array[index, :, 2])
                if r + g + b == 0:
                    tiles_scores_array[alto_slide][ancho_slide] = 0
                    break # Comienza un nuevo valor de @ancho_slide
                """ Se realiza lo mismo que se ha realizado con las filas, pero esta vez con las columnas """
                r = np.sum(sub_img_array[:, index, 0])
                g = np.sum(sub_img_array[:, index, 1])
                b = np.sum(sub_img_array[:, index, 2])
                if r + g + b == 0:
                    tiles_scores_array[alto_slide][ancho_slide] = 0
                    break # Comienza un nuevo valor de @ancho_slide
            """ Aunque estas imágenes que tienen líneas enteramente negras (ya sea horizontalmente o verticalmente) son
            leídas, al realizar la máscara del mapa de calor van a ser ocultadas, puesto que se les ha hecho que su
            puntuación sea cero. """

            """ Ahora se lee de nuevo cada tesela de 210x210, convirtiéndolas en un array para pasarlas de formato RGBA 
            a formato RGB con OpenCV. A partir de aquí, se expande la dimensión de la tesela para poder realizarle la
            predicción """
            sub_img = np.array(wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                               (ancho, alto)))
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGBA2RGB)
            tile = np.expand_dims(sub_img, axis = 0)

            """ Se va guardando la predicción para cada tesela de todos los genes en una lista. Además, para cada una de 
            ellas se guarda la puntuación de las predicciones de los genes CNV-A que interesan en el 'array' 
            correspondiente del 'array' 3D definido anteriormente, para así poder realizar despues los mapas de calor de 
            todos esos genes """
            prediction_cnv_a = model.predict(tile)[1]
            cnv_a.append(prediction_cnv_a)
            cnv_a_scores[0][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_MYC_AMP')]
            cnv_a_scores[1][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_CCND1_AMP')]
            cnv_a_scores[2][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_FGF19_AMP')]
            cnv_a_scores[3][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_ERBB2_AMP')]
            cnv_a_scores[4][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_FGF3_AMP')]

""" La lista 'tiles_scores_list' es una lista 3D donde se almacenan las puntuaciones del color de las teselas. En esta 
ocasión, la lista tiene una forma de (1, alto, ancho), es decir, hay 1 matriz con las puntuaciones de las teselas en 
forma de lista, siendo ésta de nº filas y nº columnas. """
tiles_scores_list.append(tiles_scores_array)

""" Se realiza la suma para cada una de las columnas de la lista de predicciones. Como resultado, se obtiene una lista
de 43 columnas y 1 sola fila, ya que se han sumado las predicciones de todas las teselas para cada gen. """
cnv_a = np.concatenate(cnv_a)
cnv_a_sum_columns = cnv_a.sum(axis = 0)

""" Se ordenan los índices de la lista resultante ordenador de mayor a menor valor, mostrando los resultados con mayor 
valor, que serán los de los genes con mayor probabilidad de mutación CNV-A """
mpm_cnv_a = np.argsort(cnv_a_sum_columns)[::-1][:5]

for (index_cnv_a, index_sort) in enumerate(mpm_cnv_a):
    label_cnv_a = "\nLa mutación CNV-A más probable es del gen {}".format(classes_cnv_a[index_sort].split('_')[1])
    print(label_cnv_a)

""" Se lee la WSI en un nivel de resolución lo suficientemente bajo para aplicarle despues el mapa de calor y lo 
suficientemente alto para que tenga un buen nivel de resolución """
#slide = np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1])) # TIFF
slide = np.array(wsi.read_region((0, 0), level_map, dimensions_map)) # MRXS
slide = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)

""" El tamaño de las figuras en Python se establece en pulgadas (inches). La fórmula para convertir de píxeles a 
pulgadas es (píxeles / dpi = pulgadas). La cantidad de píxeles permanece invariable porque depende de la imagen de menor
resolucion de la WSI. Por lo tanto, para convertir a pulgadas, hay que variar el DPI (puntos por pulgada) y las propias
pulgadas. """

""" --------------------------------------------------- Gen MYC ---------------------------------------------------- """
grid = cnv_a_scores[0] # (nº filas, nº columnas)
pixeles_x = slide.shape[1]
pixeles_y = slide.shape[0]
dpi = 100

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se crea una máscara para las puntuaciones menores de 0.1 y mayores de 0.8, de forma que no se pasan datos en 
aquellas celdas donde se superan dichas puntuaciones """
mask = np.zeros_like(tiles_scores_list[0])
mask[np.where(tiles_scores_list[0] < 0.2) and np.where(tiles_scores_list[0] > 0.9)] = True

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.8,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
#heatmap.imshow(np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1])),
               #aspect = heatmap.get_aspect(), extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # TIFF
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('cnv_a_MYC.png')
#plt.show()

""" -------------------------------------------- Gen CCND1 --------------------------------------------------------- """
grid = cnv_a_scores[1] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.8,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('cnv_a_CCND1.png')

""" ---------------------------------------------- Gen FGF19 ------------------------------------------------------- """
grid = cnv_a_scores[2] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.8,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('cnv_a_FGF19.png')

""" --------------------------------------------- Gen ERBB2 -------------------------------------------------------- """
grid = cnv_a_scores[3] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.8,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('cnv_a_ERBB2.png')

""" --------------------------------------------- Gen FGF3 -------------------------------------------------------- """
grid = cnv_a_scores[4] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.8,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('cnv_a_FGF3.png')