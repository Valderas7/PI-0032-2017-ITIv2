import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from tensorflow.keras.models import load_model
import seaborn as sns

""" Se carga el Excel de INiBICA y se recopilan las salidas de los genes para cada tipo de mutación """
data_inibica = pd.read_excel('/home/avalderas/img_slides/excel_genesOCA&inibica_patients/inference_inibica.xlsx',
                              engine='openpyxl')

# SNV
test_labels_snv = data_inibica.iloc[:, 16:167]

# CNV-A
test_labels_cnv_a = data_inibica.iloc[:, 167::3]

# CNV-D
test_labels_cnv_d = data_inibica.iloc[:, 169::3]

""" Ademas se recopilan los nombres de las columnas para usarlos posteriormente """
# SNV
test_columns_snv = test_labels_snv.columns.values
classes_snv = test_columns_snv.tolist()

# CNV-A
test_columns_cnv_a = test_labels_cnv_a.columns.values
classes_cnv_a = test_columns_cnv_a.tolist()

# CNV-D
test_columns_cnv_d = test_labels_cnv_d.columns.values
classes_cnv_d = test_columns_cnv_d.tolist()

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
puntuaciones de color de cada tesela """
tiles_scores_array = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Se crea una lista para recopilar las predicciones de las mutaciones y un 'array' en 3D para recopilar las 
puntuaciones de las distintas mutaciones de los genes para ese tipo de mutación. """
# SNV
snv = []
snv_genes = 7 # PIK3CA, TP53, AKT1, PTEN, ERBB2, EGFR, MTOR.
snv_scores = np.zeros((snv_genes, int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

# CNV-A
cnv_a = []
cnv_a_genes = 6 # MYC, CCND1, CDKN1B, FGF19, ERBB2, FGF3.
cnv_a_scores = np.zeros((cnv_a_genes, int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

# CNV-D
cnv_d = []
cnv_d_genes = 6 # BRCA1, BRCA2, KDR, CHEK1, FGF3, FANCA.
cnv_d_scores = np.zeros((cnv_d_genes, int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

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

            """ Se va guardando la predicción SNV, CNV-A y CNV-D de todos los genes para cada tesela en su lista 
            correspondiente. Además, para cada una de las teselas, se guarda la puntuación de las predicciones de las
            mutaciones SNV, CNV-A y CNV-D de los genes que se quieren estudiar. Estas puntuaciones para cada gen se 
            guardan dentro del 'array' correspondiente del 'array' 3D definido para cada tipo de mutación, para así 
            poder realizar después los mapas de calor de todos esos genes que interesa estudiar """
            prediction_snv = model.predict(tile)[0]     # SNV
            prediction_cnv_a = model.predict(tile)[1]   # CNV-A
            prediction_cnv_d = model.predict(tile)[3]   # CNV-D

            snv.append(prediction_snv)
            cnv_a.append(prediction_cnv_a)
            cnv_d.append(prediction_cnv_d)

            snv_scores[0][alto_slide][ancho_slide] = prediction_snv[:, classes_snv.index('SNV_PIK3CA')]
            snv_scores[1][alto_slide][ancho_slide] = prediction_snv[:, classes_snv.index('SNV_TP53')]
            snv_scores[2][alto_slide][ancho_slide] = prediction_snv[:, classes_snv.index('SNV_AKT1')]
            snv_scores[3][alto_slide][ancho_slide] = prediction_snv[:, classes_snv.index('SNV_PTEN')]
            snv_scores[4][alto_slide][ancho_slide] = prediction_snv[:, classes_snv.index('SNV_ERBB2')]
            snv_scores[5][alto_slide][ancho_slide] = prediction_snv[:, classes_snv.index('SNV_EGFR')]
            snv_scores[6][alto_slide][ancho_slide] = prediction_snv[:, classes_snv.index('SNV_MTOR')]

            cnv_a_scores[0][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_MYC_AMP')]
            cnv_a_scores[1][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_CCND1_AMP')]
            cnv_a_scores[2][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_CDKN1B_AMP')]
            cnv_a_scores[3][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_FGF19_AMP')]
            cnv_a_scores[4][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_ERBB2_AMP')]
            cnv_a_scores[5][alto_slide][ancho_slide] = prediction_cnv_a[:, classes_cnv_a.index('CNV_FGF3_AMP')]

            cnv_d_scores[0][alto_slide][ancho_slide] = prediction_cnv_d[:, classes_cnv_d.index('CNV_BRCA1_DEL')]
            cnv_d_scores[1][alto_slide][ancho_slide] = prediction_cnv_d[:, classes_cnv_d.index('CNV_BRCA2_DEL')]
            cnv_d_scores[2][alto_slide][ancho_slide] = prediction_cnv_d[:, classes_cnv_d.index('CNV_KDR_DEL')]
            cnv_d_scores[3][alto_slide][ancho_slide] = prediction_cnv_d[:, classes_cnv_d.index('CNV_CHEK1_DEL')]
            cnv_d_scores[4][alto_slide][ancho_slide] = prediction_cnv_d[:, classes_cnv_d.index('CNV_FGF3_DEL')]
            cnv_d_scores[5][alto_slide][ancho_slide] = prediction_cnv_d[:, classes_cnv_d.index('CNV_FANCA_DEL')]

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

for (index, sorted_index) in enumerate(max_snv):
    label_snv = "La mutación SNV más probable es del gen {}".format(classes_snv[sorted_index].split('_')[1])
    print(label_snv)

# CNV-A
max_cnv_a = np.argsort(cnv_a_sum_columns)[::-1][:1]

for (index, sorted_index) in enumerate(max_cnv_a):
    label_cnv_a = "La mutación CNV-A más probable es del gen {}".format(classes_cnv_a[sorted_index].split('_')[1])
    print(label_cnv_a)

# CNV-D
max_cnv_d = np.argsort(cnv_a_sum_columns)[::-1][:1]

for (index, sorted_index) in enumerate(max_cnv_d):
    label_cnv_d = "La mutación CNV-D más probable es del gen {}".format(classes_cnv_d[sorted_index].split('_')[1])
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

""" --------------------------------------------------- SNV -------------------------------------------------------- 
-------------------------------------------------------------------------------------------------------------------- """

""" -------------------------------------------------- PIK3CA ------------------------------------------------------ """
grid = snv_scores[0] # (nº filas, nº columnas)

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se crea una máscara para las puntuaciones menores de 0.1 y mayores de 0.8, de forma que no se pasan datos en 
aquellas celdas donde se superan dichas puntuaciones """
mask = np.zeros_like(tiles_scores_array)
mask[np.where(tiles_scores_array < 0.2) and np.where(tiles_scores_array > 0.9)] = True

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.8,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
#heatmap.imshow(np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1])),
               #aspect = heatmap.get_aspect(), extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # TIFF
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('snv_PIK3CA.png')
#plt.show()

""" --------------------------------------------------- TP53 ------------------------------------------------------- """
grid = snv_scores[1] # (nº filas, nº columnas)

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
plt.savefig('snv_TP53.png')

""" --------------------------------------------------- AKT1 ------------------------------------------------------- """
grid = snv_scores[2] # (nº filas, nº columnas)

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
plt.savefig('snv_AKT1.png')

""" ------------------------------------------------- PTEN --------------------------------------------------------- """
grid = snv_scores[3] # (nº filas, nº columnas)

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
plt.savefig('snv_PTEN.png')

""" ------------------------------------------------- ERBB2 --------------------------------------------------------- """
grid = snv_scores[4] # (nº filas, nº columnas)

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
plt.savefig('snv_ERBB2.png')

""" ------------------------------------------------- EGFR --------------------------------------------------------- """
grid = snv_scores[5] # (nº filas, nº columnas)

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
plt.savefig('snv_EGFR.png')

""" ------------------------------------------------- MTOR --------------------------------------------------------- """
grid = snv_scores[6] # (nº filas, nº columnas)

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
plt.savefig('snv_MTOR.png')


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
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.8,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('cnv_a_MYC.png')

""" ----------------------------------------------- CCND1 ---------------------------------------------------------- """
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

""" ----------------------------------------------- CDKN1B --------------------------------------------------------- """
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
plt.savefig('cnv_a_CDKN1B.png')

""" ------------------------------------------------ FGF19 --------------------------------------------------------- """
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
plt.savefig('cnv_a_FGF19.png')

""" --------------------------------------------- ERBB2 ------------------------------------------------------------ """
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
plt.savefig('cnv_a_ERBB2.png')

""" ------------------------------------------------ FGF3 ---------------------------------------------------------- """
grid = cnv_a_scores[5] # (nº filas, nº columnas)

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
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = False, cmap = "Reds", alpha = 0.8,
                      zorder = 2, vmin = 0.0, vmax = 1.0)

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS
plt.savefig('cnv_d_BRCA1.png')

""" ----------------------------------------------- BRCA2 ---------------------------------------------------------- """
grid = cnv_d_scores[1] # (nº filas, nº columnas)

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
plt.savefig('cnv_d_BRCA2.png')

""" ----------------------------------------------- KDR ----------------------------------------------------------- """
grid = cnv_d_scores[2] # (nº filas, nº columnas)

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
plt.savefig('cnv_d_KDR.png')

""" ------------------------------------------------ CHEK1 --------------------------------------------------------- """
grid = cnv_d_scores[3] # (nº filas, nº columnas)

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
plt.savefig('cnv_d_CHEK1.png')

""" ---------------------------------------------- FGF3 ------------------------------------------------------------ """
grid = cnv_d_scores[4] # (nº filas, nº columnas)

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
plt.savefig('cnv_d_FGF3.png')

""" ------------------------------------------------ FANCA --------------------------------------------------------- """
grid = cnv_d_scores[5] # (nº filas, nº columnas)

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
plt.savefig('cnv_d_FANCA.png')
