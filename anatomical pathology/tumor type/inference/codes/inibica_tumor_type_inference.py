""" Programa para hacer una clasificación por teselas de las WSI (.mrxs) del proyecto. Se crean mapas de calor con las
predicciones de tipo histológico en tumores mixtos.

Table 2. Tumor type samples in TCGA dataset with lymph node patients

|       Tumor Type  |
|   Type  | Samples |
    IDC       105
    ILC       105
Mucinous        6
Medillary       1
Metaplastic     1
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

""" Se carga el Excel de INiBICA ya transformado para variables anatomopatológicas, y se recopilan las salidas """
data_inibica = pd.read_excel('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/excel/inference_inibica_mutations-anatomopathologic.xlsx',
                              engine='openpyxl')

test_labels_tumor_type = data_inibica.iloc[:, 238:245]

""" Parámetros de las teselas """
ancho = 210
alto = 210
canales = 3

""" Se carga el modelo de la red neuronal """
path = '/anatomical pathology/tumor type/inference/models/model_image_tumor_type_06_0.47_ultimate.h5'
model = load_model(path)

""" Se abre WSI especificada y extraemos el paciente del que se trata """
path_wsi = '/media/proyectobdpath/PI0032WEB/P029-HE-097-3_v2.mrxs'
wsi = openslide.OpenSlide(path_wsi)
patient_id = path_wsi.split('/')[4][:4]

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
puntuaciones de color de cada tesela """
tiles_scores_array = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Se crea una lista y un array 3D para recopilar las predicciones y las puntuaciones, respectivamente, de la 
predicción del tipo histológico. """
tumor_type_list = []
tumor_type_scores = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

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

        if 0.1 <= tiles_scores_array[alto_slide][ancho_slide] < 0.9:
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
                        break # Salta a la línea #123
                else:
                    continue
                break # Salta a la línea #131
            """ Aunque estas imágenes que tienen líneas enteramente negras (ya sea horizontalmente o verticalmente) son
            leídas, al realizar la máscara del mapa de calor van a ser ocultadas, puesto que se les ha hecho que su
            puntuación sea uno. """

            """ Ahora se lee de nuevo cada tesela de 210x210, convirtiéndolas en un array para pasarlas de formato RGBA 
            a formato RGB con OpenCV. A partir de aquí, se expande la dimensión de la tesela para poder realizarle la
            predicción """
            if 0.1 <= tiles_scores_array[alto_slide][ancho_slide] < 0.9:
                sub_img = np.array(wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                               (ancho, alto)))
                sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGBA2RGB)
                tile = np.expand_dims(sub_img, axis = 0)

                """ Se va guardando la predicción de los datos anatomopatológicos para cada tesela en su lista 
                correspondiente. Además, para cada una de las teselas, se guarda el índice más alto de las predicciones 
                de todos los datos anatomopatológicos. Estos índices se guardan dentro del 'array' correspondiente del 
                'array' 3D definido para cada tipo de dato anatomopatológico, para así poder realizar después los mapas 
                de calor de todos los datos """
                prediction_tumor_type = model.predict(tile)     # Tipo histológico
                tumor_type_list.append(prediction_tumor_type)
                idc_ilc_selection = np.take(prediction_tumor_type, [0, 1]) # Elegir solo IDC(0) o ILC(1) en tipos mixtos
                tumor_type_scores[alto_slide][ancho_slide] = np.argmax(idc_ilc_selection)

""" Se realiza la suma de las columnas para cada una de las predicciones de cada datos anatomopatológicos. Como 
resultado, se obtiene un array de varias columnas (dependiendo del dato anatomopatológico habrá más o menos clases) y 
una sola fila, ya que se han sumado las predicciones de todas las teselas. Este array se ordena por los índice de mayor 
a menor puntuación, siendo el de mayor puntuación la predicción de la clase del dato anatomopatológico analizado """
# Tipo histológico
tumor_type_classes = ['Invasive Ductal Carcinoma', 'Invasive Lobular Carcinoma', 'Medullary', 'Metaplastic', 'Mucinous']

tumor_type = np.concatenate(tumor_type_list)
tumor_type_sum = np.array(tumor_type.sum(axis = 0))
max_tumor_type = int(np.argsort(tumor_type_sum)[::-1][:1])
print("Tipo histológico: {}".format(tumor_type_classes[max_tumor_type]))

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
---------------------------------------------------- Tipo histológico --------------------------------------------------
------------------------------------------------------------------------------------------------------------------------ """
grid = tumor_type_scores # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se crea una máscara para las puntuaciones menores de 0.09 y mayores de 0.9, de forma que no se pasan datos en 
aquellas celdas donde se superan dichas puntuaciones """
mask = np.zeros_like(tiles_scores_array)
mask[np.where((tiles_scores_array <= 0.1) | (tiles_scores_array > 0.9))] = True

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = True,
                      cmap = LinearSegmentedColormap.from_list('Custom', ((0.8, 0, 0, 1), (0, 0, 0.8, 1)), 2),
                      alpha = 0.2, zorder = 2, cbar_kws = {'shrink': 0.2}, yticklabels = False, xticklabels = False)

""" Se edita la barra leyenda del mapa de calor para que muestre los nombres de las categorías de los tipos histológicos
y no números. """
colorbar = heatmap.collections[0].colorbar
colorbar.set_ticks([0.75, 0.25])
colorbar.set_ticklabels(['Ductal', 'Lobular'])

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
#heatmap.imshow(np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1])),
               #aspect = heatmap.get_aspect(), extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # TIFF
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS

""" Se guarda el mapa de calor, eliminando el espacio blanco que sobra en los ejes X e Y de la imagen """
plt.savefig('/home/avalderas/img_slides/anatomical pathology/image/tumor type/inference/heatmaps/tumor type {}.png'.format(patient_id),
            bbox_inches = 'tight')
#plt.show()