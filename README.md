# PI-0032-2017
En este repositorio se encuentran datos, archivos y códigos desarrollados para el proyecto `PI0032-2017`, titulado `Predicción de riesgo de metástasis a distancia del carcinoma de mama mediante la integración de datos morfológicos, imunohistoquímicos y genéticos, a través de tecnología big data`.

------------

##### Resumen:
- **cnv&snv_mutations**: En esta carpeta se encuentran los códigos de entrenamiento e inferencia de predicciones de mutaciones `SNV` y de mutaciones `CNV-A` y `CNV-D`.

- **correlations**: Aquí se encuentran los códigos de entrenamiento e inferencia de la correlación entre distintos grupos de datos (datos clínicos vs datos anatomopatológicos; datos anatomopatológicos vs mutaciones, etc).

- **data**: En este directorio se encuentran los datos recogidos de la `API` de cBioPortal y que servirán como punto de partida para desarrollar los distintos códigos de entrenamiento.

- **excel_genesOCA&inibica_patients**: En esta carpeta se encuentran dos documentos en `Excel`. Uno de ellos muestra cuales son los distintos genes del `panel OCA` y en qué categorías entran; y el otro muestra información de distintas variables clínicas, anatomopatológicas y de inmunohistoquímica de cada uno de todos los pacientes de `INiBICA` a los que se les realiza la inferencia de los distintos códigos desarrollados.

- **img_lotes**: Aquí se encuentran todas las imágenes estáticas a las que se les ha realizado una captura de pantalla para entrenar las redes neuronales convolucionales. 

- **patient_status**: En este directorio se encuentran los códigos de entrenamiento e inferencia para la predicción de las variables clínicas más importantes: supervivencia general del paciente, recidivas y metástasis a distancia.

- **screenshots**: En esta carpeta se recopilan capturas de pantalla que muestran distintos resultados (matriz de confusión, métricas, etc.) de las distintas redes neuronales realizadas.

- **split_image_into_tiles**: Aquí se almacenan tanto los códigos para dividir las imágenes de la carpeta `img_lotes` en teselas, como las propias teselas.

- **subtype_classification**: En este directorio se encuentran los códigos de entrenamiento e inferencia para la predicción de subtipos moleculares (Luminal A, Luminal B, Basal, etc).