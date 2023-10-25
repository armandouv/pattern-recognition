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
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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

# Procesa cada imagen en el directorio de entrada
for image_file in os.listdir(input_directory):
    image_path = os.path.join(input_directory, image_file)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    height, width, _ = original_image.shape

    output_image_kmeans = np.zeros((height, width, 3), dtype=np.uint8)
    output_image_naive_bayes = np.zeros((height, width, 3), dtype=np.uint8)
    output_image_knn = np.zeros((height, width, 3), dtype=np.uint8)

    window_size = 8  # Tamaño de la ventana deslizante

    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            print("pixel ", y, x)
            window = original_image[y:y+window_size, x:x+window_size]

            glcm_features = calculate_glcm_features(window, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])

            label_kmeans = kmeans.predict([glcm_features])[0]
            color_kmeans = color_mapping[label_kmeans]

            label_naive_bayes = naive_bayes.predict([glcm_features])[0]
            color_naive_bayes = color_mapping[label_naive_bayes]

            label_knn = knn.predict([glcm_features])[0]
            color_knn = color_mapping[label_knn]

            output_image_kmeans[y:y+window_size, x:x+window_size] = color_kmeans
            output_image_naive_bayes[y:y+window_size, x:x+window_size] = color_naive_bayes
            output_image_knn[y:y+window_size, x:x+window_size] = color_knn

    # Guarda las imágenes de salida en el directorio de resultados
    filename, _ = os.path.splitext(image_file)
    cv2.imwrite(os.path.join(output_directory, f'{filename}_kmeans.png'), output_image_kmeans)
    cv2.imwrite(os.path.join(output_directory, f'{filename}_naive_bayes.png'), output_image_naive_bayes)
    cv2.imwrite(os.path.join(output_directory, f'{filename}_knn.png'), output_image_knn)
