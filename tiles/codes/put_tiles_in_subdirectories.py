import shelve # datos persistentes
import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
import glob
import shutil

""" Directorios de teselas con cancer """
image_dir = '/tiles/TCGA_no_normalizadas_cáncer'

""" Se seleccionan todas las rutas de las imágenes que tienen cáncer: """
cancer_dir = glob.glob(image_dir + "/*.JPG")

for image in cancer_dir:
    if int(image.split('_')[9][3:]) < 100:
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles1/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 100) and (int(image.split('_')[9][3:]) < 200):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles2/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 200) and (int(image.split('_')[9][3:]) < 300):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles3/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 300) and (int(image.split('_')[9][3:]) < 400):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles4/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 400) and (int(image.split('_')[9][3:]) < 500):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles5/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 500) and (int(image.split('_')[9][3:]) < 600):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles6/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 600) and (int(image.split('_')[9][3:]) < 700):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles7/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 700) and (int(image.split('_')[9][3:]) < 800):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles8/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 800) and (int(image.split('_')[9][3:]) < 900):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles9/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 900) and (int(image.split('_')[9][3:]) < 1000):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles10/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 1000) and (int(image.split('_')[9][3:]) < 1100):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles11/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 1100) and (int(image.split('_')[9][3:]) < 1200):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles12/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 1200) and (int(image.split('_')[9][3:]) < 1300):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles13/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 1300) and (int(image.split('_')[9][3:]) < 1400):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles14/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 1400) and (int(image.split('_')[9][3:]) < 1500):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles15/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 1500) and (int(image.split('_')[9][3:]) < 1600):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles16/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 1600) and (int(image.split('_')[9][3:]) < 1700):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles17/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 1700) and (int(image.split('_')[9][3:]) < 1800):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles18/' + image[79:])
    if (int(image.split('_')[9][3:]) >= 1800) and (int(image.split('_')[9][3:]) < 1900):
        shutil.move(image, '/home/avalderas/img_slides/split_images_into_tiles/TCGA_no_normalizadas_cáncer/img_lotes_tiles19/' + image[79:])
