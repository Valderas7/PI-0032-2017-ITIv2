""" Programa para hacer una clasificación por teselas de las WSI (.mrxs) del proyecto. Se crean mapas de calor con las
predicciones de supervivencia. """

""" Se importan librerías """
import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from tensorflow.keras.models import load_model
import seaborn as sns
import staintools
from sklearn.metrics import confusion_matrix
import itertools

""" Parámetros de las teselas """
ancho = 210
alto = 210
canales = 3

""" Se carga el modelo de la red neuronal """
path = '/clinical/image/distant metastasis/inference/models/model_image_metastasis_39_0.35_normalized.h5'
model = load_model(path)

""" Se abre WSI especificada y extraemos el paciente del que se trata """
path_wsi = '/media/proyectobdpath/PI0032WEB/P008-HE-141-A2_v2.mrxs'
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

""" Se añade el método de normalización 'vahadane' que es el que ha sido usado en el proceso de entrenamiento. Se ajusta
este método con la misma imagen que se usó en el proceso de entrenamiento """
target = staintools.read_image('/home/avalderas/img_slides/images/img_lote1_cancer/TCGA-A2-A25D-01Z-00-DX1.2.JPG')
target = staintools.LuminosityStandardizer.standardize(target)
normalizer = staintools.StainNormalizer(method = 'vahadane')
normalizer.fit(target)

""" Se crea un 'array' con forma (alto, ancho), que son el número de filas y el número de columnas, respectivamente, en 
el que se divide la WSI al dividirla en teselas de 210x210 en el nivel de resolucion máximo, para recopilar asi las 
puntuaciones de color de cada tesela """
tiles_scores_array = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Se crea una lista y un array 3D para recopilar las predicciones y las puntuaciones, respectivamente, de la 
predicción del tipo histológico. """
metastasis_list = []
metastasis_scores = np.zeros((int(dim[1]/(alto * scale)), int(dim[0] / (ancho * scale))))

""" Además, se crean otra lista para ir recopilando las teselas de la WSI, las clases verdaderas de cada tesela y las
predicciones de cada tesela. """
test_image_data = []
test_labels_true = []
test_labels_prediction = []

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
                        break # Salta a la línea #113
                else:
                    continue
                break # Salta a la línea #121
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
                #sub_img = staintools.LuminosityStandardizer.standardize(sub_img)
                #sub_img = normalizer.transform(sub_img)
                test_image_data.append(sub_img)
                tile = np.expand_dims(sub_img, axis = 0)
                test_labels_true.append(1) # Porque P008 tiene metástasis a distancia

                """ Se va guardando la predicción de los datos anatomopatológicos para cada tesela en su lista 
                correspondiente. Además, para cada una de las teselas, se guarda el índice más alto de las predicciones 
                de todos los datos anatomopatológicos. Estos índices se guardan dentro del 'array' correspondiente del 
                'array' 3D definido para cada tipo de dato anatomopatológico, para así poder realizar después los mapas 
                de calor de todos los datos """
                prediction_metastasis = model.predict(tile) # [[Número]]
                metastasis_list.append(prediction_metastasis)
                metastasis_scores[alto_slide][ancho_slide] = prediction_metastasis # Número
                test_labels_prediction.append(metastasis_scores[alto_slide][ancho_slide])

""" Se realiza la suma de las columnas para cada una de las predicciones de cada datos anatomopatológicos. Como 
resultado, se obtiene un array de varias columnas (dependiendo del dato anatomopatológico habrá más o menos clases) y 
una sola fila, ya que se han sumado las predicciones de todas las teselas. Este array se ordena por los índice de mayor 
a menor puntuación, siendo el de mayor puntuación la predicción de la clase del dato anatomopatológico analizado """
#tumor_type = np.concatenate(tumor_type_list)
#tumor_type_sum = np.array(tumor_type.sum(axis = 0))
#max_tumor_type = int(np.argsort(tumor_type_sum)[::-1][:1])
#print("Tipo histológico: {}".format(tumor_type_classes[max_tumor_type]))

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

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. """
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
test_labels_metastasis = np.asarray(test_labels_true)
test_image_data = np.asarray(test_image_data)

results = model.evaluate(test_image_data, test_labels_metastasis, verbose = 0)

""" -------------------------------------------------------------------------------------------------------------------
------------------------------------------- SECCIÓN DE EVALUACIÓN  ----------------------------------------------------
--------------------------------------------------------------------------------------------------------------------"""
# Loss
print("\n'Loss' de metástasis a distancia en el conjunto de prueba: {:.2f}".format(results[0]))

# Sensibilidad
if results[1] + results[4] > 0:
    recall = results[5] * 100
    print("Sensibilidad de metástasis a distancia en el conjunto de prueba: {:.2f}%".format(recall))
else:
    recall = "No definido"
    print("Sensibilidad de metástasis a distancia en el conjunto de prueba: {}".format(recall))

# Precisión
if results[1] + results[2] > 0:
    precision = results[6] * 100
    print("Precisión de metástasis a distancia en el conjunto de prueba: {:.2f}%".format(precision))
else:
    precision = "No definido"
    print("Precisión de metástasis a distancia en el conjunto de prueba: {}".format(precision))

# Valor-F
if results[5] > 0 or results[6] > 0:
    print("Valor-F de metástasis a distancia en el conjunto de "
          "prueba: {:.2f}".format((2 * results[5] * results[6]) / (results[5] + results[6])))

# Especificidad
if results[2] + results[3] > 0:
    specifity = (results[3]/(results[3]+results[2])) * 100
    print("Especificidad de metástasis a distancia en el conjunto de prueba: {:.2f}%".format(specifity))
else:
    specifity = "No definido"
    print("Especifidad de metástasis a distancia en el conjunto de prueba: {}".format(specifity))

# Exactitud
accuracy = results[7]
print("Exactitud de metástasis a distancia en el conjunto de prueba: {:.2f}%".format(accuracy * 100))

# Curvas ROC
if (len(np.unique(test_labels_metastasis))) > 1:
    print("AUC-ROC de metástasis a distancia en el conjunto de prueba: {:.2f}".format(results[8]))
    print("AUC-PR de metástasis a distancia en el conjunto de prueba: {:.2f}".format(results[9]))

    """ Se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
    documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y 
    la tasa de falsos positivos, por lo que resume el comportamiento general del clasificador para diferenciar clases: """
    # @ravel: Aplana el vector a 1D
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    y_pred_prob_metastasis = model.predict(test_image_data).ravel()
    fpr, tpr, thresholds = roc_curve(test_labels_metastasis, y_pred_prob_metastasis)
    auc_roc = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc_roc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC-ROC curve for distant metastasis prediction')
    plt.legend(loc='best')
    plt.show()

    """ Por otra parte, tambien se dibuja el area bajo la la curva PR (precision-recall), para tener un documento grafico 
    del rendimiento del clasificador en cuanto a la sensibilidad y la precision de resultados. """
    precision, recall, threshold_metastasis = precision_recall_curve(test_labels_metastasis, y_pred_prob_metastasis)
    auc_pr = auc(recall, precision)

    plt.figure(2)
    plt.plot([0, 1], [0, 0], 'k--', label='No Skill')
    plt.plot(recall, precision, label='AUC = {:.2f})'.format(auc_pr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR curve for distant metastasis prediction')
    plt.legend(loc='best')
    plt.show()

""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos.
Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Matriz de confusión', cmap = plt.cm.Blues):
    """ Imprime y dibuja la matriz de confusión. Se puede normalizar escribiendo el parámetro `normalize=True`. """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
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
        plt.text(j, il, cm[il, j], horizontalalignment = "center", color = "white" if cm[il, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Clase verdadera')
    plt.xlabel('Predicción')

# Metástasis
y_true_metastasis = test_labels_metastasis
y_pred_metastasis = np.round(np.array(test_labels_prediction))

matrix_metastasis = confusion_matrix(y_true_metastasis, y_pred_metastasis, labels = [0, 1])
matrix_metastasis_classes = ['Sin metástasis distante', 'Con metástasis distante']

plot_confusion_matrix(matrix_metastasis, classes = matrix_metastasis_classes, title ='Matriz de confusión de metástasis '
                                                                                     'a distancia')
plt.show()

""" -------------------------------------------------------------------------------------------------------------------- 
-------------------------------------------------- Metástasis a distancia ----------------------------------------------
------------------------------------------------------------------------------------------------------------------------ """
grid = metastasis_scores

""" Se reescala el mapa de calor que se va a implementar posteriormente a las dimensiones de la imagen de mínima 
resolución del WSI """
sns.set(style = "white", rc = {'figure.dpi': dpi})
plt.subplots(figsize = (pixeles_x/dpi, pixeles_y/dpi))
plt.tight_layout()

""" Se crea una máscara para las puntuaciones menores de 0.09 y mayores de 0.9, de forma que no se pasan datos en 
aquellas celdas donde se superan dichas puntuaciones """
mask = np.zeros_like(tiles_scores_array)
mask[np.where((tiles_scores_array <= 0.1) | (tiles_scores_array > 0.9) | (metastasis_scores > 0.5))] = True

""" Se dibuja el mapa de calor """
heatmap = sns.heatmap(grid, square = True, linewidths = .5, mask = mask, cbar = True, cmap = 'Reds',
                      alpha = 0.5, zorder = 2, cbar_kws = {'shrink': 0.2}, yticklabels = False, xticklabels = False)

""" Se edita la barra leyenda del mapa de calor para que muestre los nombres de las categorías de los tipos histológicos
y no números. """
colorbar = heatmap.collections[0].colorbar
colorbar.set_ticks([0.75, 0.25]) # Esto hay que cambiarlo dependiendo de los valores máximos y mínimos del mapa de calor
colorbar.set_ticklabels(['>Metástasis', '<Metástasis'])

""" Se adapta la imagen de mínima resolución del WSI a las dimensiones del mapa de calor (que anteriormente fue
redimensionado a las dimensiones de la imagen de mínima resolución del WSI) """
#heatmap.imshow(np.array(wsi.read_region((0, 0), wsi.level_count - 1, wsi.level_dimensions[wsi.level_count - 1])),
               #aspect = heatmap.get_aspect(), extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # TIFF
heatmap.imshow(np.array(wsi.read_region((0, 0), level_map, dimensions_map)), aspect = heatmap.get_aspect(),
               extent = heatmap.get_xlim() + heatmap.get_ylim(), zorder = 1) # MRXS

""" Se guarda el mapa de calor, eliminando el espacio blanco que sobra en los ejes X e Y de la imagen """
plt.savefig('/home/avalderas/img_slides/clinical/image/distant metastasis/inference/heatmaps/metastasis_{}.png'.format(patient_id),
            bbox_inches = 'tight')
#plt.show()