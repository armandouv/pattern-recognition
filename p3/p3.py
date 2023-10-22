import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import graycomatrix, graycoprops

# Directorio de datos
data_directory = 'p3/entrenamiento'
image_folders = os.listdir(data_directory)
classes = ['mono', 'tronco', 'hojas']
class_labels = {c: i for i, c in enumerate(classes)}

# Función para calcular características de textura (GLCM)
def calculate_glcm_features(image, distances, angles, log=False):
    # Calcula la matriz GLCM
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)
    
    # Calcula estadísticos de segundo orden
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]

    if log:
        print(f'Contrast: {contrast}')
        print(f'Homogeneity: {homogeneity}')
        print(f'Energy: {energy}')
        print(f'Correlation: {correlation}')
        print(f'Dissimilarity: {dissimilarity}')
        print(f'ASM: {ASM}')
    
    return np.array([contrast, homogeneity, energy, correlation, dissimilarity, ASM])

# Preprocesamiento de datos
data = []
labels = []

for folder in image_folders:
    class_label = class_labels[folder]
    folder_path = os.path.join(data_directory, folder)
    image_files = os.listdir(folder_path)
    print(f'Clase: {classes[class_label]}')
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        
        print(f'Imagen: {i + 1}')
        # Calcula características de textura (GLCM) y agrega a los datos
        glcm_features = calculate_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], log=True)
        data.append(glcm_features)
        labels.append(class_label)

# Aplica K-Means
kmeans = KMeans(n_clusters=len(classes))
kmeans.fit(data)

# Inicializa y entrena los clasificadores Naive Bayes y K-NN
naive_bayes = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)  # Experimenta con el número de vecinos
naive_bayes.fit(data, labels)
knn.fit(data, labels)

# Asigna colores a las etiquetas de K-Means
color_mapping = {
    0: (0, 0, 255),  # Rojo
    1: (0, 255, 0),  # Verde
    2: (255, 0, 0)   # Azul
}

# Directorio de entrada y salida
input_directory = 'p3/prueba'  # Directorio con imágenes a clasificar
output_directory = 'p3/resultados'  # Directorio para guardar las imágenes de salida
# Cargar la imagen de entrada
image_file = os.listdir(input_directory)[0]
image_path = os.path.join(input_directory, image_file)
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
height, width, _ = original_image.shape

# Configurar el número de hilos según el número de núcleos disponibles (4 en este caso)
num_threads = 4

# Dividir la imagen en partes iguales
part_height = height // num_threads

import concurrent.futures

window_size = 32

# Función para procesar una parte de la imagen
def process_image_part(part_start, part_end, kmeans, naive_bayes, knn, color_mapping):
    part_output_image_kmeans = np.zeros((part_end - part_start, width, 3), dtype=np.uint8)
    part_output_image_naive_bayes = np.zeros((part_end - part_start, width, 3), dtype=np.uint8)
    part_output_image_knn = np.zeros((part_end - part_start, width, 3), dtype=np.uint8)

    for y in range(part_start, part_end):
        for x in range(0, width):
            print("pixel ", y, x)
            y_start = max(0, y - window_size // 2)
            y_end = min(height, y + window_size // 2 + 1)
            x_start = max(0, x - window_size // 2)
            x_end = min(width, x + window_size // 2 + 1)
            window = original_image[y_start:y_end, x_start:x_end]

            glcm_features = calculate_glcm_features(window, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])

            label_kmeans = kmeans.predict([glcm_features])[0]
            color_kmeans = color_mapping[label_kmeans]

            label_naive_bayes = naive_bayes.predict([glcm_features])[0]
            color_naive_bayes = color_mapping[label_naive_bayes]

            label_knn = knn.predict([glcm_features])[0]
            color_knn = color_mapping[label_knn]

            part_output_image_kmeans[y - part_start, x] = color_kmeans
            part_output_image_naive_bayes[y - part_start, x] = color_naive_bayes
            part_output_image_knn[y - part_start, x] = color_knn

    return part_output_image_kmeans, part_output_image_naive_bayes, part_output_image_knn

# Procesar partes de la imagen en paralelo
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []

    for i in range(num_threads):
        part_start = i * part_height
        part_end = (i + 1) * part_height if i < num_threads - 1 else height
        futures.append(executor.submit(process_image_part, part_start, part_end, kmeans, naive_bayes, knn, color_mapping))

    output_image_kmeans = np.vstack([f.result()[0] for f in futures])
    output_image_naive_bayes = np.vstack([f.result()[1] for f in futures])
    output_image_knn = np.vstack([f.result()[2] for f in futures])

# Guardar las imágenes de salida
filename, _ = os.path.splitext(image_file)
cv2.imwrite(os.path.join(output_directory, f'{filename}_kmeans.png'), output_image_kmeans)
cv2.imwrite(os.path.join(output_directory, f'{filename}_naive_bayes.png'), output_image_naive_bayes)
cv2.imwrite(os.path.join(output_directory, f'{filename}_knn.png'), output_image_knn)
