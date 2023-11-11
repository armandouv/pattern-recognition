import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation

# Ruta de la imagen de entrada
input_image_path = 'p3/prueba/n0110.jpg'

# Directorio de salida para los superpíxeles SLIC
output_directory = 'p3/entrenamiento/slic_superpixels'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Número deseado de superpíxeles
num_superpixels = 100

# Carga la imagen de entrada
image = io.imread(input_image_path)

# Realiza la segmentación de superpíxeles con SLIC
labels = segmentation.slic(image, n_segments=num_superpixels, compactness=10)

# Crea una máscara para superponer los límites de los superpíxeles
boundary_mask = segmentation.mark_boundaries(image, labels, color=(1, 1, 0))

# Convierte la máscara a uint8 antes de guardarla
boundary_mask_uint8 = (boundary_mask * 255).astype(np.uint8)

# Guarda la imagen resultante con los límites de los superpíxeles
output_filename = os.path.join(output_directory, 'slic_superpixels_result.png')
io.imsave(output_filename, boundary_mask_uint8)

print(f'Imagen resultante de superpíxeles SLIC guardada en {output_filename}.')

# Muestra la imagen resultante
plt.imshow(boundary_mask)
plt.title('Superpíxeles SLIC')
plt.show()
