""" Programa para hacer una clasificación por teselas de las WSI (.mrxs) del proyecto. Se crean mapas de calor con las
predicciones de la mutación objetivo.
                                                MUTADOS TP53
P002    P004    P008    P009    P011    P012    P013    P016    P017    P020    P024    P031    P032    P034    P038
P047    P048    P051    P056    P057    P059    P077    P078    P079    P084    P086    P090    P092    P105    P107
P124    P126    P150    P154    P168    P169    P170    P178    P189    P195    P197    P202
"""

""" Se importan librerías """
import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from tensorflow.keras.models import load_model
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import staintools

""" Parámetros de las teselas """
ancho = 210
alto = 210
canales = 3

""" Se carga el modelo de la red neuronal """
path = '/home/avalderas/img_slides/mutations/image/SNV/TP53/inference/models/model_image_tp53_02_0.59.h5'
model = load_model(path)

""" Se abre WSI especificada y extraemos el paciente del que se trata """
path_wsi = '/media/proyectobdpath/PI0032WEB/P002-HE-033-2_v2.mrxs'
wsi = openslide.OpenSlide(path_wsi)
patient_id = path_wsi.split('/')[4][:4]

"""" Se hallan las dimensiones (anchura, altura) del nivel de resolución '0' (máxima resolución) de la WSI """
dim = wsi.dimensions

""" Se averigua cual es el mejor nivel de resolución de la WSI para mostrar el factor de reduccion deseado. Las WSI de
los pacientes de la provincia de Cádiz tienen su nivel de máxima resolución a una magnificación de 40x. Por lo que si
se aplica un factor de reducción de 4, se hallará el mejor nivel para una magnificación 10x (40x/4 = 10x), que es la que 
interesa buscar puesto que es la magnificación con la que se entrenó la red neuronal. """
best_level = wsi.get_best_level_for_downsample(4)

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

""" Se añade el método de normalización 'vahadane' que es el que ha sido usado en el proceso de entrenamiento. Se ajusta
este método con la misma imagen que se usó en el proceso de entrenamiento """
target = staintools.read_image('/home/avalderas/img_slides/images/img_lote1_cancer/TCGA-A2-A25D-01Z-00-DX1.2.JPG')
target = staintools.LuminosityStandardizer.standardize(target)
normalizer = staintools.StainNormalizer(method = 'vahadane')
normalizer.fit(target)

""" Se crea un 'array' con forma (alto, ancho), que son el número de filas y el número de columnas, respectivamente, en 
el que se divide la WSI al dividirla en teselas de 210x210 en el nivel de resolucion máximo, para recopilar asi las 
puntuaciones de color (blanco o negro) de cada tesela """
tiles_scores_array = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Se crea una lista y un array 3D para recopilar las predicciones y las puntuaciones, respectivamente, de la 
predicción de la mutación. """
mutation_list = []
mutation_scores = np.zeros((int(dim[1] / (alto * scale)), int(dim[0] / (ancho * scale))))

""" Se itera sobre todas las teselas de tamaño 210x210 de la WSI en el nivel adecuado al factor de reduccion '10x' """
#@ancho_slide itera de (0 - nºcolumnas) [columnas] y @alto_slide de (0 - nºfilas) [filas]
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

        if 0.10 <= tiles_scores_array[alto_slide][ancho_slide] < 0.7:
            """ Primero se intenta hallar si hay una línea recta negra que dura todo el ancho de la tesela. Para ello se
            itera sobre todas las filas de los tres canales RGB de la tesela para saber si en algún momento la suma de 
            tres filas correspodientes en los tres canales de la tesela es cero, lo que indicaría que hay una fila 
            entera en la tesela de color negro, y por tanto, se hace que su puntuacion sea cero. """
            sub_img_array = cv2.cvtColor(np.array(sub_img), cv2.COLOR_RGBA2RGB)
            r_row = 1; g_row = 2; b_row = 3; r_col= 1; g_col = 2; b_col = 3
            for index_row in range(210):
                for index_col in range(210):
                    r_row = int(sub_img_array[index_row, index_col, 0])
                    g_row = int(sub_img_array[index_row, index_col, 1])
                    b_row = int(sub_img_array[index_row, index_col, 2])
                    r_col = int(sub_img_array[index_col, index_row, 0])
                    g_col = int(sub_img_array[index_col, index_row, 1])
                    b_col = int(sub_img_array[index_col, index_row, 2])
                    if (r_row + g_row + b_row == 0) | (r_col + g_col + b_col == 0):
                        tiles_scores_array[alto_slide][ancho_slide] = 1.0
                        break # Salta a la línea #118
                else:
                    continue
                break # Salta a la línea #126
            """ Aunque estas imágenes que tienen líneas enteramente negras (ya sea horizontalmente o verticalmente) son
            leídas, al realizar la máscara del mapa de calor van a ser ocultadas, puesto que se les ha hecho que su
            puntuación sea uno. """

            """ Ahora se lee de nuevo cada tesela de 210x210, convirtiéndolas en un array para pasarlas de formato RGBA 
            a formato RGB con OpenCV. A partir de aquí, se expande la dimensión de la tesela para poder realizarle la
            predicción """
            if 0.10 <= tiles_scores_array[alto_slide][ancho_slide] < 0.7:
                sub_img = np.array(wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                               (ancho, alto)))
                sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGBA2RGB)
                sub_img = staintools.LuminosityStandardizer.standardize(sub_img)
                sub_img = normalizer.transform(sub_img)
                #cv2.imshow('tile', sub_img)
                #cv2.waitKey(0)
                tile = np.expand_dims(sub_img, axis = 0)

                """ Se va guardando la predicción de los datos anatomopatológicos para cada tesela en su lista 
                correspondiente. Además, para cada una de las teselas, se guarda el índice más alto de las predicciones 
                de todos los datos anatomopatológicos. Estos índices se guardan dentro del 'array' correspondiente del 
                'array' 3D definido para cada tipo de dato anatomopatológico, para así poder realizar después los mapas 
                de calor de todos los datos """
                prediction_mutation = model.predict(tile)
                mutation_scores[alto_slide][ancho_slide] = prediction_mutation # Puntuación rango (0,1) para mapa calor
                mutation_list.append(np.round(mutation_scores[alto_slide][ancho_slide])) # Lista de 0s y 1s para probab.

""" Se realiza una cuenta de las teselas con y sin mutación. Se divide el número de teselas con mutación entre el 
número total de teselas y se multiplica posteriormente por cien, para obtener el procentaje de probabilidad de que+
exista mutación en la WSI. """
# Mutación
mutation_classes = ['Con mutación', 'Sin mutación']
overall_probability_prediction = mutation_list.count(1.0) / (mutation_list.count(0.0) + mutation_list.count(1.0))
print("Probabilidad de encontrar este gen mutado: {:.4}%".format(overall_probability_prediction * 100))

""" Se lee la WSI en un nivel de resolución lo suficientemente bajo para aplicarle después el mapa de calor y lo 
suficientemente alto para que tenga un buen nivel de resolución """
#slide = np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1])) # TIFF
slide = np.array(wsi.read_region((0, 0), level_map, dimensions_map)) # MRXS
slide = cv2.cvtColor(slide, cv2.COLOR_RGBA2RGB)

""" El tamaño de las figuras en Python se establece en pulgadas (inches). La fórmula para convertir de píxeles a 
pulgadas es (píxeles / dpi = pulgadas). La cantidad de píxeles permanece invariable porque depende de la imagen de menor
resolucion de la WSI. Por lo tanto, para convertir a pulgadas, hay que variar el DPI (puntos por pulgada) y las propias
pulgadas. """
pixeles_x = slide.shape[1]
pixeles_y = slide.shape[0]
dpi = 100

""" -------------------------------------------------------------------------------------------------------------------- 
------------------------------------------------- Mapa de calor --------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------ """
grid = mutation_scores # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se crea una máscara para las puntuaciones menores de 0.09 y mayores de 0.9, de forma que no se pasan datos en 
aquellas celdas donde se superan dichas puntuaciones """
mask = np.zeros_like(tiles_scores_array)
mask[np.where((tiles_scores_array <= 0.1) | (tiles_scores_array > 0.9)) and np.where(mutation_scores <= 0.5)] = True

""" Implementando colores del mapa de calor """
c = ["whitesmoke", "yellow", "darkorange", "red", "darkred"]
v = [0, 0.5, 0.7, 0.9, 1]
l = list(zip(v,c))
cmap = LinearSegmentedColormap.from_list('mutation_map', l, N = 256)

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = True, cmap = cmap, alpha = 0.3,
                      zorder = 2, cbar_kws = {'shrink': 0.2}, yticklabels = False, xticklabels = False)

""" Se edita la barra leyenda del mapa de calor para que muestre los nombres de las categorías de los tipos histológicos
y no números. """
colorbar = heatmap.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['Sin mutación', 'Con mutación'])

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
#heatmap.imshow(np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1])),
               #aspect = heatmap.get_aspect(), extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # TIFF
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS

""" Se guarda el mapa de calor, eliminando el espacio blanco que sobra en los ejes X e Y de la imagen """
plt.savefig('/home/avalderas/img_slides/mutations/image/SNV/TP53/inference/heatmaps/SNV_TP53_{}.png'.format(patient_id),
            bbox_inches = 'tight')
#plt.show()