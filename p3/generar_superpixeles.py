import numpy as np
from skimage import io, segmentation, color
from matplotlib import pyplot as plt

# Carga la imagen
image = io.imread('prueba/n0110.jpg')

# Convierte la imagen a Lab color para una mejor segmentación
lab_image = color.rgb2lab(image)

# Aplica SLIC para obtener superpíxeles
segments1 = segmentation.slic(lab_image, n_segments=10, compactness=10)
segments2 = segmentation.slic(lab_image, n_segments=50, compactness=10)
segments3 = segmentation.slic(lab_image, n_segments=100, compactness=10)

# Crea un superpíxel y grafica la imagen segmentada
out10 = color.label2rgb(segments1, image, kind='avg')
out50 = color.label2rgb(segments2, image, kind='avg') 
out100 = color.label2rgb(segments3, image, kind='avg')

plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(out10)
plt.title('10 Superpíxeles')

plt.subplot(132)
plt.imshow(out50)
plt.title('50 Superpíxeles')

plt.subplot(133)
plt.imshow(out100)
plt.title('100 Superpíxeles')

plt.show()
