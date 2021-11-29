""" Programa para hacer una clasificación por teselas de las WSI (.mrxs) del proyecto. Se crean mapas de calor con las
predicciones para cada uno de los datos de anatomía patológica.

Table 2. Anatomical pathology samples in TCGA dataset with lymph node patients (552 patients)

|       Tumor Type  |       STAGE      |      pT      |       pN      |      pM       |       IHQ       |
|   Type  | Samples | STAGE  | Samples | pT | Samples | pN  | Samples | pM |  Samples | Subtype | Samples |
    IDC         402    IIB       214    T2      331     N1A     164     M0      442      Lum-A      254
    ILC         105    IIIA      155    T3       94     N1      123     MX       92      Lum-B      113
    Other       20     IIA        67    T1C      79     N2A      64     M1       18      Basal      64
Mixed(NOS)      17     IIIC       64    T4B      20     N2       55                      Her2       46
Mucinous        6      IIIB       19    T1       13     N3A      46                      Normal     16
Medillary       1      IV         16    T4        6     N1MI     36
Metaplastic     1      IB          6    T1B       3     N1B      32
                       II          3    T4D       3     N3       26
                       X           3    T1A       1     N3B       3
                       III         2    T2B       1     N1C       2
                                                        N3C       1
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

""" Se carga el Excel de INiBICA ya transformado para variables anatomopatológicas, y se recopilan las salidas """
data_inibica = pd.read_excel('/home/avalderas/img_slides/correlations/mutations-anatomopathologic/inference/test_data&models/inference_inibica_mutations-anatomopathologic.xlsx',
                              engine='openpyxl')

test_labels_tumor_type = data_inibica.iloc[:, 238:245]
test_labels_STAGE = data_inibica.iloc[:, 245:255]
test_labels_pT = data_inibica.iloc[:, 255:265]
test_labels_pN = data_inibica.iloc[:, 265:276]
test_labels_pM = data_inibica.iloc[:, 276:279]
test_labels_IHQ = data_inibica.iloc[:, 279:]

""" Parámetros de las teselas """
ancho = 210
alto = 210
canales = 3

""" Se carga el modelo de la red neuronal """
path = '/home/avalderas/img_slides/anatomical_pathology_data/image/inference/models/model_image_anatomical_pathology.h5'
model = load_model(path)
epoch_model = path.split('_')[4]

""" Se abre WSI especificada """
path_wsi = '/media/proyectobdpath/PI0032WEB/P002-HE-033-2_v2.mrxs'
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
puntuaciones de color de cada tesela """
tiles_scores_array = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Se crea una lista de seis listas (una para cada uno de los datos anatomopatológicos) y un array 3D para recopilar
los índices de cada uno de los datos anatomopatológicos. """
anatomical_pathology_list = []
anatomical_pathology_data = 6

for i in range(anatomical_pathology_data):
    anatomical_pathology_list.append([])

anatomical_pathology_data_scores = np.zeros((anatomical_pathology_data, int(dim[1]/(alto * scale)),
                                             int(dim[0] / (ancho * scale))))

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

        if 0.09 < tiles_scores_array[alto_slide][ancho_slide] < 0.9:
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
                    tiles_scores_array[alto_slide][ancho_slide] = 1.0
                    break # Salta a la línea #164
                """ Se realiza lo mismo que se ha realizado con las filas, pero esta vez con las columnas """
                r = np.sum(sub_img_array[:, index, 0])
                g = np.sum(sub_img_array[:, index, 1])
                b = np.sum(sub_img_array[:, index, 2])
                if r + g + b == 0:
                    tiles_scores_array[alto_slide][ancho_slide] = 1.0
                    break # Salta a la línea #164
            """ Aunque estas imágenes que tienen líneas enteramente negras (ya sea horizontalmente o verticalmente) son
            leídas, al realizar la máscara del mapa de calor van a ser ocultadas, puesto que se les ha hecho que su
            puntuación sea uno. """

            """ Ahora se lee de nuevo cada tesela de 210x210, convirtiéndolas en un array para pasarlas de formato RGBA 
            a formato RGB con OpenCV. A partir de aquí, se expande la dimensión de la tesela para poder realizarle la
            predicción """
            if 0.09 < tiles_scores_array[alto_slide][ancho_slide] < 0.9:
                sub_img = np.array(wsi.read_region((ancho_slide * (210 * scale), alto_slide * (210 * scale)), best_level,
                                               (ancho, alto)))
                sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGBA2RGB)
                tile = np.expand_dims(sub_img, axis = 0)

                """ Se va guardando la predicción de los datos anatomopatológicos para cada tesela en su lista 
                correspondiente. Además, para cada una de las teselas, se guarda el índice más alto de las predicciones 
                de todos los datos anatomopatológicos. Estos índices se guardan dentro del 'array' correspondiente del 
                'array' 3D definido para cada tipo de dato anatomopatológico, para así poder realizar después los mapas 
                de calor de todos los datos """
                prediction_tumor_type = model.predict(tile)[0]      # Tipo histológico
                prediction_STAGE = model.predict(tile)[1]           # STAGE
                prediction_pT = model.predict(tile)[2]              # pT
                prediction_pN = model.predict(tile)[3]              # pN
                prediction_pM = model.predict(tile)[4]              # pM
                prediction_IHQ = model.predict(tile)[5]             # IHQ

                anatomical_pathology_list[0].append(prediction_tumor_type)
                anatomical_pathology_list[1].append(prediction_STAGE)
                anatomical_pathology_list[2].append(prediction_pT)
                anatomical_pathology_list[3].append(prediction_pN)
                anatomical_pathology_list[4].append(prediction_pM)
                anatomical_pathology_list[5].append(prediction_IHQ)

                anatomical_pathology_data_scores[0][alto_slide][ancho_slide] = np.argmax(prediction_tumor_type)
                anatomical_pathology_data_scores[1][alto_slide][ancho_slide] = np.argmax(prediction_STAGE)
                anatomical_pathology_data_scores[2][alto_slide][ancho_slide] = np.argmax(prediction_pT)
                anatomical_pathology_data_scores[3][alto_slide][ancho_slide] = np.argmax(prediction_pN)
                anatomical_pathology_data_scores[4][alto_slide][ancho_slide] = np.argmax(prediction_pM)
                anatomical_pathology_data_scores[5][alto_slide][ancho_slide] = np.argmax(prediction_IHQ)

""" Se realiza la suma para cada una de las columnas de la lista de predicciones. Como resultado, se obtiene una lista
de (genes) columnas y 1 sola fila, ya que se han sumado las predicciones de todas las teselas para cada gen. """
# SNV
snv = np.concatenate(snv)
snv_sum_columns = snv.sum(axis = 0)

# CNV-A
cnv_a = np.concatenate(cnv_a)
cnv_a_sum_columns = cnv_a.sum(axis = 0)

# CNV-D
cnv_d = np.concatenate(cnv_d)
cnv_d_sum_columns = cnv_d.sum(axis = 0)

""" Se ordenan los índices de la lista resultante ordenador de mayor a menor valor, mostrando los resultados con mayor 
valor, que serán los de los genes con mayor probabilidad de mutación """
# SNV
max_snv = np.argsort(snv_sum_columns)[::-1][:1]

for (index, sorted_index_snv) in enumerate(max_snv):
    label_snv = "La mutación SNV más probable es del gen {}".format(classes_snv[sorted_index_snv].split('_')[1])
    print(label_snv)

# CNV-A
max_cnv_a = np.argsort(cnv_a_sum_columns)[::-1][:1]

for (index, sorted_index_cnv_a) in enumerate(max_cnv_a):
    label_cnv_a = "La mutación CNV-A más probable es del gen {}".format(classes_cnv_a[sorted_index_cnv_a].split('_')[1])
    print(label_cnv_a)

# CNV-D
max_cnv_d = np.argsort(cnv_d_sum_columns)[::-1][:1]

for (index, sorted_index_cnv_d) in enumerate(max_cnv_d):
    label_cnv_d = "La mutación CNV-D más probable es del gen {}".format(classes_cnv_d[sorted_index_cnv_d].split('_')[1])
    print(label_cnv_d)

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

""" Se crean las carpetas para guardar los mapas de calor que se corresponden con la 'epoch' del modelo elegido """
new_snv_epoch_folder = '/home/avalderas/img_slides/mutations/image/inference/heatmaps/SNV/Epoch{}'.format(epoch_model)
new_cnv_a_epoch_folder = '/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-A/Epoch{}'.format(epoch_model)
new_cnv_d_epoch_folder = '/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-D/Epoch{}'.format(epoch_model)

if not os.path.exists(new_snv_epoch_folder):
    os.makedirs(new_snv_epoch_folder)
if not os.path.exists(new_cnv_a_epoch_folder):
    os.makedirs(new_cnv_a_epoch_folder)
if not os.path.exists(new_cnv_d_epoch_folder):
    os.makedirs(new_cnv_d_epoch_folder)

""" --------------------------------------------------- SNV -------------------------------------------------------- 
-------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------- PIK3CA ------------------------------------------------------ """
grid = snv_scores[0] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se crea una máscara para las puntuaciones menores de 0.09 y mayores de 0.9, de forma que no se pasan datos en 
aquellas celdas donde se superan dichas puntuaciones """
mask = np.zeros_like(tiles_scores_array)
mask[np.where((tiles_scores_array < 0.09) | (tiles_scores_array > 0.9))] = True

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
#heatmap.imshow(np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1])),
               #aspect = heatmap.get_aspect(), extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # TIFF
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/SNV/Epoch{}/snv_PIK3CA.png'.format(epoch_model))
#plt.show()

""" --------------------------------------------------- TP53 ------------------------------------------------------- """
grid = snv_scores[1] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/SNV/Epoch{}/snv_TP53.png'.format(epoch_model))

""" --------------------------------------------------- AKT1 ------------------------------------------------------- """
grid = snv_scores[2] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/SNV/Epoch{}/snv_AKT1.png'.format(epoch_model))

""" ------------------------------------------------- PTEN --------------------------------------------------------- """
grid = snv_scores[3] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/SNV/Epoch{}/snv_PTEN.png'.format(epoch_model))

""" ------------------------------------------------- ERBB2 --------------------------------------------------------- """
grid = snv_scores[4] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/SNV/Epoch{}/snv_ERBB2.png'.format(epoch_model))

""" ------------------------------------------------- EGFR --------------------------------------------------------- """
grid = snv_scores[5] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/SNV/Epoch{}/snv_EGFR.png'.format(epoch_model))

""" ------------------------------------------------- MTOR --------------------------------------------------------- """
grid = snv_scores[6] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/SNV/Epoch{}/snv_MTOR.png'.format(epoch_model))


""" ------------------------------------------------- CNV-A -------------------------------------------------------- 
-------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------- MYC --------------------------------------------------------- """
grid = cnv_a_scores[0] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-A/Epoch{}/cnv_a_MYC.png'.format(epoch_model))

""" ----------------------------------------------- CCND1 ---------------------------------------------------------- """
grid = cnv_a_scores[1] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-A/Epoch{}/cnv_a_CCND1.png'.format(epoch_model))

""" ----------------------------------------------- CDKN1B --------------------------------------------------------- """
grid = cnv_a_scores[2] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-A/Epoch{}/cnv_a_CDKN1B.png'.format(epoch_model))

""" ------------------------------------------------ FGF19 --------------------------------------------------------- """
grid = cnv_a_scores[3] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-A/Epoch{}/cnv_a_FGF19.png'.format(epoch_model))

""" --------------------------------------------- ERBB2 ------------------------------------------------------------ """
grid = cnv_a_scores[4] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-A/Epoch{}/cnv_a_ERBB2.png'.format(epoch_model))

""" ------------------------------------------------ FGF3 ---------------------------------------------------------- """
grid = cnv_a_scores[5] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-A/Epoch{}/cnv_a_FGF3.png'.format(epoch_model))

""" ------------------------------------------------- CNV-D -------------------------------------------------------- 
-------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------- BRCA1 -------------------------------------------------------- """
grid = cnv_d_scores[0] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-D/Epoch{}/cnv_d_BRCA1.png'.format(epoch_model))

""" ----------------------------------------------- BRCA2 ---------------------------------------------------------- """
grid = cnv_d_scores[1] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-D/Epoch{}/cnv_d_BRCA2.png'.format(epoch_model))

""" ----------------------------------------------- KDR ----------------------------------------------------------- """
grid = cnv_d_scores[2] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-D/Epoch{}/cnv_d_KDR.png'.format(epoch_model))

""" ------------------------------------------------ CHEK1 --------------------------------------------------------- """
grid = cnv_d_scores[3] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-D/Epoch{}/cnv_d_CHEK1.png'.format(epoch_model))

""" ---------------------------------------------- FGF3 ------------------------------------------------------------ """
grid = cnv_d_scores[4] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-D/Epoch{}/cnv_d_FGF3.png'.format(epoch_model))

""" ------------------------------------------------ FANCA --------------------------------------------------------- """
grid = cnv_d_scores[5] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.5,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('/home/avalderas/img_slides/mutations/image/inference/heatmaps/CNV-D/Epoch{}/cnv_d_FANCA.png'.format(epoch_model))