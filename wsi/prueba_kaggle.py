import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os
from tensorflow.keras.models import load_model
import matplotlib.ticker as plticker

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

# Open
biopsy = openslide.OpenSlide(path_wsi)
level = biopsy.level_count - 1
dimensions = biopsy.level_dimensions[level]
sample = biopsy.read_region((0, 0), level, dimensions)

# Resolution
dpi = 100

""" Se averigua cual es el factor de reduccion del nivel que mejor representa las imagenes entrenadas. Se multiplica 
este factor de reduccion por las dimensiones del nivel de resolucion maximo, que es el nivel de referencia para la 
funcion @read_region """
scale_10 = int(wsi.level_downsamples[best_level])
scale_max_level = int(wsi.level_downsamples[level])

# Set up figure
fig = plt.figure(figsize=(float(sample.size[0]) / dpi, float(sample.size[1]) / dpi), dpi=dpi)
ax = fig.add_subplot(111)

# Set the gridding interval: here we use the major tick interval
interval = ancho / scale_max_level
loc = plticker.MultipleLocator(base=interval)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

# Add the image
ax.imshow(sample)

# Add the grid
ax.grid(which='major', axis='both', linestyle='-')

# Find number of gridsquares in x and y direction
nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(interval)))
ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(interval)))

# Add some labels to the gridsquares
for i in range(nx):
    x = interval / 2. + float(i) * interval
    ax.text(x, interval / 2, i, color='black', ha='center', va='center')
for j in range(ny):
    y = interval / 2 + j * interval
    ax.text(interval / 2, y, j, color='black', ha='center', va='center')

# Save the figure
fig.savefig('myImageGrid.tiff', dpi = dpi)

# Close
biopsy.close()
sample = None

