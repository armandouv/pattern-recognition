import cv2
import os

# Ruta de la imagen de entrada
input_image_path = 'p3/prueba/n0110.jpg'

# Directorio de salida para las ventanas
output_directory = 'p3/entrenamiento/ventanas'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Tamaño de las ventanas cuadradas
window_size = 8

# Número deseado de ventanas
num_windows = 100

# Carga la imagen de entrada
image = cv2.imread(input_image_path)

# Obtén las dimensiones de la imagen
height, width, _ = image.shape

# Calcula la separación entre ventanas
step = max(height, width) // int(num_windows**0.5)

# Contador para el nombre de las ventanas
window_count = 0

# Genera y guarda las ventanas
for y in range(0, height - window_size + 1, step):
    for x in range(0, width - window_size + 1, step):
        if window_count < num_windows:
            # Extrae la ventana
            window = image[y:y+window_size, x:x+window_size]

            # Genera el nombre de archivo para la ventana
            window_filename = os.path.join(output_directory, f'window_{window_count}.jpg')

            # Guarda la ventana como una imagen individual
            cv2.imwrite(window_filename, window)

            window_count += 1

print(f'Se han guardado {window_count} ventanas en el directorio {output_directory}.')
