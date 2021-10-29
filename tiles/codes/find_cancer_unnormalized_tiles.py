""" Para seleccionar las imagenes no normalizadas con cáncer a partir de las imagenes normalizadas con cancer.
Se comparan los nombres de los archivos de ambas carpetas y se seleccionan solo las coincidentes. """
import cv2 #OpenCV
import glob
import os
from shutil import copyfile

""" Listado de archivos de teselas normalizadas con cáncer: """
normal_tile1 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles1')
normal_tile2 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles2')
normal_tile3 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles3')
normal_tile4 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles4')
normal_tile5 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles5')
normal_tile6 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles6')
normal_tile7 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles7')
normal_tile8 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles8')
normal_tile9 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles9')
normal_tile10 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles10')
normal_tile11 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles11')
normal_tile12 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles12')
normal_tile13 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles13')
normal_tile14 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles14')
normal_tile15 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles15')
normal_tile16 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles16')
normal_tile17 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles17')
normal_tile18 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_normalizadas_cáncer/img_lotes_tiles18')

""" Listado de directorios de teselas no normalizadas con y sin cáncer: """
dir_1 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_1'
dir_2 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_2'
dir_3 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_3'
dir_4 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_4'
dir_5 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_5'
dir_6 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_6'
dir_7 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_7'
dir_8 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_8'
dir_9 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_9'
dir_10 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_10'
dir_11 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_11'
dir_12 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_12'
dir_13 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_13'
dir_14 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_14'
dir_15 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_15'
dir_16 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_16'
dir_17 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_17'
dir_18 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_18'
dir_19 = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_19'

""" Listado de archivos de teselas no normalizadas con y sin cáncer: """
unnormal_tile1 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_1')
unnormal_tile2 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_2')
unnormal_tile3 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_3')
unnormal_tile4 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_4')
unnormal_tile5 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_5')
unnormal_tile6 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_6')
unnormal_tile7 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_7')
unnormal_tile8 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_8')
unnormal_tile9 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_9')
unnormal_tile10 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_10')
unnormal_tile11 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_11')
unnormal_tile12 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_12')
unnormal_tile13 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_13')
unnormal_tile14 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_14')
unnormal_tile15 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_15')
unnormal_tile16 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_16')
unnormal_tile17 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_17')
unnormal_tile18 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_18')
unnormal_tile19 = os.listdir('/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_sin_y_con_cáncer/img_lotes_tiles_un_19')

common_files_1 = set(unnormal_tile1).intersection(normal_tile1)
common_files_2 = set(unnormal_tile1).intersection(normal_tile3)
common_files_3 = set(unnormal_tile2).intersection(normal_tile1)
common_files_4 = set(unnormal_tile2).intersection(normal_tile2)
common_files_5 = set(unnormal_tile2).intersection(normal_tile3)
common_files_6 = set(unnormal_tile3).intersection(normal_tile2)
common_files_7 = set(unnormal_tile3).intersection(normal_tile3)
common_files_8 = set(unnormal_tile3).intersection(normal_tile4)
common_files_9 = set(unnormal_tile3).intersection(normal_tile5)
common_files_10 = set(unnormal_tile4).intersection(normal_tile3)
common_files_11 = set(unnormal_tile4).intersection(normal_tile4)
common_files_12 = set(unnormal_tile4).intersection(normal_tile5)
common_files_13 = set(unnormal_tile5).intersection(normal_tile3)
common_files_14 = set(unnormal_tile5).intersection(normal_tile4)
common_files_15 = set(unnormal_tile5).intersection(normal_tile5)
common_files_16 = set(unnormal_tile5).intersection(normal_tile6)
common_files_17 = set(unnormal_tile6).intersection(normal_tile5)
common_files_18 = set(unnormal_tile6).intersection(normal_tile6)
common_files_19 = set(unnormal_tile6).intersection(normal_tile7)
common_files_20 = set(unnormal_tile7).intersection(normal_tile6)
common_files_21 = set(unnormal_tile7).intersection(normal_tile7)
common_files_22 = set(unnormal_tile8).intersection(normal_tile7)
common_files_23 = set(unnormal_tile8).intersection(normal_tile8)
common_files_24 = set(unnormal_tile9).intersection(normal_tile8)
common_files_25 = set(unnormal_tile9).intersection(normal_tile9)
common_files_26 = set(unnormal_tile10).intersection(normal_tile9)
common_files_27 = set(unnormal_tile10).intersection(normal_tile10)
common_files_28 = set(unnormal_tile11).intersection(normal_tile10)
common_files_29 = set(unnormal_tile11).intersection(normal_tile11)
common_files_30 = set(unnormal_tile12).intersection(normal_tile11)
common_files_31 = set(unnormal_tile12).intersection(normal_tile12)
common_files_32 = set(unnormal_tile13).intersection(normal_tile12)
common_files_33 = set(unnormal_tile13).intersection(normal_tile13)
common_files_34 = set(unnormal_tile13).intersection(normal_tile14)
common_files_35 = set(unnormal_tile14).intersection(normal_tile13)
common_files_36 = set(unnormal_tile14).intersection(normal_tile14)
common_files_37 = set(unnormal_tile15).intersection(normal_tile14)
common_files_38 = set(unnormal_tile15).intersection(normal_tile15)
common_files_39 = set(unnormal_tile15).intersection(normal_tile16)
common_files_40 = set(unnormal_tile16).intersection(normal_tile14)
common_files_41 = set(unnormal_tile16).intersection(normal_tile15)
common_files_42 = set(unnormal_tile16).intersection(normal_tile16)
common_files_43 = set(unnormal_tile17).intersection(normal_tile16)
common_files_44 = set(unnormal_tile17).intersection(normal_tile17)
common_files_45 = set(unnormal_tile18).intersection(normal_tile17)
common_files_46 = set(unnormal_tile18).intersection(normal_tile18)
common_files_47 = set(unnormal_tile19).intersection(normal_tile18)

""" Directorio destino de las teselas coincidentes """
output_dir = '/home/avalderas/img_slides/tiles/TCGA_no_normalizadas_cáncer'

for file in common_files_1:
    copyfile(os.path.join(dir_1, file),
             os.path.join(output_dir, file))

for file in common_files_2:
    copyfile(os.path.join(dir_1, file),
             os.path.join(output_dir, file))

for file in common_files_3:
    copyfile(os.path.join(dir_2, file),
             os.path.join(output_dir, file))

for file in common_files_4:
    copyfile(os.path.join(dir_2, file),
             os.path.join(output_dir, file))

for file in common_files_5:
    copyfile(os.path.join(dir_2, file),
             os.path.join(output_dir, file))

for file in common_files_6:
    copyfile(os.path.join(dir_3, file),
             os.path.join(output_dir, file))

for file in common_files_7:
    copyfile(os.path.join(dir_3, file),
             os.path.join(output_dir, file))

for file in common_files_8:
    copyfile(os.path.join(dir_3, file),
             os.path.join(output_dir, file))

for file in common_files_9:
    copyfile(os.path.join(dir_3, file),
             os.path.join(output_dir, file))

for file in common_files_10:
    copyfile(os.path.join(dir_4, file),
             os.path.join(output_dir, file))

for file in common_files_11:
    copyfile(os.path.join(dir_4, file),
             os.path.join(output_dir, file))

for file in common_files_12:
    copyfile(os.path.join(dir_4, file),
             os.path.join(output_dir, file))

for file in common_files_13:
    copyfile(os.path.join(dir_5, file),
             os.path.join(output_dir, file))

for file in common_files_14:
    copyfile(os.path.join(dir_5, file),
             os.path.join(output_dir, file))

for file in common_files_15:
    copyfile(os.path.join(dir_5, file),
             os.path.join(output_dir, file))

for file in common_files_16:
    copyfile(os.path.join(dir_5, file),
             os.path.join(output_dir, file))

for file in common_files_17:
    copyfile(os.path.join(dir_6, file),
             os.path.join(output_dir, file))

for file in common_files_18:
    copyfile(os.path.join(dir_6, file),
             os.path.join(output_dir, file))

for file in common_files_19:
    copyfile(os.path.join(dir_6, file),
             os.path.join(output_dir, file))

for file in common_files_20:
    copyfile(os.path.join(dir_7, file),
             os.path.join(output_dir, file))

for file in common_files_21:
    copyfile(os.path.join(dir_7, file),
             os.path.join(output_dir, file))

for file in common_files_22:
    copyfile(os.path.join(dir_8, file),
             os.path.join(output_dir, file))

for file in common_files_23:
    copyfile(os.path.join(dir_8, file),
             os.path.join(output_dir, file))

for file in common_files_24:
    copyfile(os.path.join(dir_9, file),
             os.path.join(output_dir, file))

for file in common_files_25:
    copyfile(os.path.join(dir_9, file),
             os.path.join(output_dir, file))

for file in common_files_26:
    copyfile(os.path.join(dir_10, file),
             os.path.join(output_dir, file))

for file in common_files_27:
    copyfile(os.path.join(dir_10, file),
             os.path.join(output_dir, file))

for file in common_files_28:
    copyfile(os.path.join(dir_11, file),
             os.path.join(output_dir, file))

for file in common_files_29:
    copyfile(os.path.join(dir_11, file),
             os.path.join(output_dir, file))

for file in common_files_30:
    copyfile(os.path.join(dir_12, file),
             os.path.join(output_dir, file))

for file in common_files_31:
    copyfile(os.path.join(dir_12, file),
             os.path.join(output_dir, file))

for file in common_files_32:
    copyfile(os.path.join(dir_13, file),
             os.path.join(output_dir, file))

for file in common_files_33:
    copyfile(os.path.join(dir_13, file),
             os.path.join(output_dir, file))

for file in common_files_34:
    copyfile(os.path.join(dir_13, file),
             os.path.join(output_dir, file))

for file in common_files_35:
    copyfile(os.path.join(dir_14, file),
             os.path.join(output_dir, file))

for file in common_files_36:
    copyfile(os.path.join(dir_14, file),
             os.path.join(output_dir, file))

for file in common_files_37:
    copyfile(os.path.join(dir_15, file),
             os.path.join(output_dir, file))

for file in common_files_38:
    copyfile(os.path.join(dir_15, file),
             os.path.join(output_dir, file))

for file in common_files_39:
    copyfile(os.path.join(dir_15, file),
             os.path.join(output_dir, file))

for file in common_files_40:
    copyfile(os.path.join(dir_16, file),
             os.path.join(output_dir, file))

for file in common_files_41:
    copyfile(os.path.join(dir_16, file),
             os.path.join(output_dir, file))

for file in common_files_42:
    copyfile(os.path.join(dir_16, file),
             os.path.join(output_dir, file))

for file in common_files_43:
    copyfile(os.path.join(dir_17, file),
             os.path.join(output_dir, file))

for file in common_files_44:
    copyfile(os.path.join(dir_17, file),
             os.path.join(output_dir, file))

for file in common_files_45:
    copyfile(os.path.join(dir_18, file),
             os.path.join(output_dir, file))

for file in common_files_46:
    copyfile(os.path.join(dir_18, file),
             os.path.join(output_dir, file))

for file in common_files_47:
    copyfile(os.path.join(dir_19, file),
             os.path.join(output_dir, file))