import shelve # datos persistentes
import pandas as pd
import numpy as np
import seaborn as sns # Para realizar gráficas sobre datos
import matplotlib.pyplot as plt
import cv2 #OpenCV
from tensorflow.keras import backend as K
import glob
import imblearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input # Para instanciar tensores de Keras
from functools import reduce # 'reduce' aplica una función pasada como argumento para todos los miembros de una lista.
from sklearn.preprocessing import MinMaxScaler # Para escalar valores
from sklearn.metrics import confusion_matrix # Para realizar la matriz de confusión
from pytorch_grad_cam import GradCAM # Hace falta instalar pytorch, ttach, torchvision, tqdm y el propio GRAD-CAM

alto = 315 # 630
ancho = 740 # 1480
canales = 3 # Imágenes a color (RGB) = 3

model = load_model('/modelos/SNV/image/model_snv_image_tp53_epoch02.h5')

test_image_data = np.load('/home/avalderas/img_slides/cnv&snv_mutations/snv/training_codes/test_image.npy')
test_labels = np.load('/home/avalderas/img_slides/cnv&snv_mutations/snv/training_codes/test_labels.npy')

"""last_conv_layer_name = 'separable_conv2d_5'
classifier_layer_names =  ['max_pooling2d_2', 'flatten', 'dense', 'dense_1']

image = cv2.imread('/home/avalderas/Descargas/ductal-carcinoma.jpg')
image = image.astype('float32') / 255
image = cv2.resize(image, (ancho, alto))
img_array = np.expand_dims(image, axis=0)

prediction = model.predict(img_array)
print(prediction)
index = np.argmax(prediction)
print([prediction[0]][index])


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # Modelo que mapea la imagen de entrada a la capa convolucional última,
    # donde se calculará la activación
    last_conv_layer = model.get_layer(last_conv_layer_name)
    conv_model = keras.Model(model.inputs, last_conv_layer.output)

    # Modelo que mapea las activaciones a la salida final
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Cálculo del gradiente la salida  del modelo clasificador respecto a
    with tf.GradientTape() as tape:

        # Calcula activacion del modelo base convolucional
        last_conv_layer_output = conv_model(img_array)
        tape.watch(last_conv_layer_output)

        # Calcula la predicción con modelo clasificador, para la clase mas probable
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        print(top_pred_index)
        top_class_channel = preds[:, top_pred_index]

    # Obtenemos el gradiente en la capa final clasificadora con respecto a
    # la salida del modelo base convolucional
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # Vector de pesos: medias del gradiente por capas,
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # salida de la última capa convolucional
    last_conv_layer_output = last_conv_layer_output.numpy()[0]

    # saliencia es la respuesta promedio de la última capa convolucional
    saliency = np.mean(last_conv_layer_output, axis=-1)
    saliency = np.maximum(saliency, 0) / np.max(saliency)

    # Multiplicación de cada canal por el vector de pesos
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Heatmap: promedio de cada canal por su peso
    grad_cam = np.mean(last_conv_layer_output, axis=-1)
    grad_cam = np.maximum(grad_cam, 0) / np.max(grad_cam)

    return grad_cam, saliency

# Generate class activation heatmap
grad_cam, saliency = make_gradcam_heatmap(img_array, 
                                          model, 
                                          last_conv_layer_name, 
                                          classifier_layer_names)

def show_hotmap(img, heatmap, title='Heatmap', alpha=0.6, cmap='jet', axisOnOff='off'):
    '''
    img     :    Image
    heatmap :    2d narray
    '''
    resized_heatmap = resize(heatmap, img.size)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(resized_heatmap, alpha=alpha, cmap=cmap)
    plt.axis(axisOnOff)
    plt.title(title)
    plt.show()

plt.subplot(121)
plt.imshow(grad_cam, 'jet')
plt.title('GradCam')
plt.subplot(122)
plt.imshow(saliency, 'jet')
plt.title('Saliencia')
plt.show()

show_hotmap(img=image, heatmap=grad_cam, title=f'Grad Cam: {model.name}')

quit()"""

""" Una vez entrenado el modelo, se puede evaluar con los datos de test y obtener los resultados de las métricas
especificadas en el proceso de entrenamiento. En este caso, se decide mostrar los resultados de la 'loss', la exactitud,
la sensibilidad y la precisión del conjunto de datos de validación."""
# @evaluate: Devuelve el valor de la 'loss' y de las métricas del modelo especificadas.
results = model.evaluate(test_image_data,test_labels, verbose = 0)
print("\n'Loss' del conjunto de prueba: {:.2f}\n""Sensibilidad del conjunto de prueba: {:.2f}\n" 
      "Precisión del conjunto de prueba: {:.2f}\n""Exactitud del conjunto de prueba: {:.2f} %\n"
      "El AUC ROC del conjunto de prueba es de: {:.2f}".format(results[0], results[5], results[6], results[7] * 100,
                                                               results[8]))

""" Por último, y una vez entrenada ya la red, también se pueden hacer predicciones con nuevos ejemplos usando el
conjunto de datos de test que se definió anteriormente al repartir los datos. """
# @suppress=True: Muestra los números con representación de coma fija
# @predict: Genera predicciones para nuevas entradas
print("\nGenera predicciones para 10 muestras")
print("Etiquetas: ", test_labels[:10])
np.set_printoptions(precision=3, suppress=True)
print("Predicciones:\n", np.round(model.predict(test_image_data[:10])))

""" Además, se realiza la matriz de confusión sobre todo el conjunto del dataset de test para evaluar la precisión de la
red neuronal y saber la cantidad de falsos positivos, falsos negativos, verdaderos negativos y verdaderos positivos. """
# @zip: Une las tuplas del nombre de los grupos con la de la cantidad de casos por grupo
y_true = test_labels # Etiquetas verdaderas de 'test'
y_pred = np.round(model.predict(test_image_data)) # Predicción de etiquetas de 'test'

matrix = confusion_matrix(y_true, y_pred) # Calcula (pero no dibuja) la matriz de confusión

group_names = ['True Neg','False Pos','False Neg','True Pos'] # Nombres de los grupos
group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()] # Cantidad de casos por grupo

labels = [f'{v1}\n{v2}\n' for v1, v2 in zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
plt.show()

""" Para finalizar, se dibuja el area bajo la curva ROC (curva caracteristica operativa del receptor) para tener un 
documento grafico del rendimiento del clasificador binario. Esta curva representa la tasa de verdaderos positivos y la
tasa de falsos positivos, por lo que resume el comportamiento del clasificador para diferenciar clases.
Para implementarla, se importan los paquetes necesarios, se definen las variables y con ellas se dibuja la curva: """
# @ravel: Aplana el vector a 1D
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_pred = y_pred.ravel()
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='AUC = {:.2f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()