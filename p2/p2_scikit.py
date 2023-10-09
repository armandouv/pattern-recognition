import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Cargar imágenes de entrenamiento para cada clase y preprocesarlas
imagenes_entrenamiento = {
    'Platano': cv2.imread('./platano.jpg', cv2.IMREAD_COLOR),
    'Huevo': cv2.imread('./huevo.jpg', cv2.IMREAD_COLOR),
    'Chile': cv2.imread('./chile.jpg', cv2.IMREAD_COLOR),
    'Fondo': cv2.imread('./fondo.jpg', cv2.IMREAD_COLOR)
}
vectores_rgb = {}
etiquetas = []

sigma = 1.0

# Aplicar un filtro gaussiano a las imágenes de entrenamiento y eliminar píxeles blancos
for clase, imagen_entrenamiento in imagenes_entrenamiento.items():
    imagen_entrenamiento_rgb = cv2.cvtColor(imagen_entrenamiento, cv2.COLOR_BGR2RGB)
    
    # Definir el color blanco en RGB
    blanco = (255, 255, 255)
    
    # Crear una máscara para los píxeles que son blancos
    mascara = cv2.inRange(imagen_entrenamiento_rgb, blanco, blanco)
    
    # Invertir la máscara para que los píxeles blancos sean 0 y los no blancos sean 1
    mascara = cv2.bitwise_not(mascara)
    
    # Aplicar la máscara a cada canal de color
    imagen_entrenamiento_sin_blanco = cv2.bitwise_and(imagen_entrenamiento_rgb, imagen_entrenamiento_rgb, mask=mascara)
    
    # Aplicar el filtro gaussiano
    imagen_entrenamiento_sin_blanco = cv2.GaussianBlur(imagen_entrenamiento_sin_blanco, (0, 0), sigma)
    
    # Aplana los valores de color de la imagen sin los píxeles blancos
    vector_r = imagen_entrenamiento_sin_blanco[:, :, 0].flatten()  # Canal Rojo
    vector_g = imagen_entrenamiento_sin_blanco[:, :, 1].flatten()  # Canal Verde
    vector_b = imagen_entrenamiento_sin_blanco[:, :, 2].flatten()  # Canal Azul
    
    indices_no_blancos = np.where(mascara.flatten() != 0)

    # Utiliza los índices para crear nuevos vectores sin los píxeles blancos
    vector_r_sin_blancos = vector_r[indices_no_blancos]
    vector_g_sin_blancos = vector_g[indices_no_blancos]
    vector_b_sin_blancos = vector_b[indices_no_blancos]

    # Almacenar la imagen preprocesada y los vectores sin blancos
    vectores_rgb[clase] = np.column_stack((vector_r_sin_blancos, vector_g_sin_blancos, vector_b_sin_blancos))
    etiquetas.extend([clase] * len(vector_r_sin_blancos))

# Convertir etiquetas a números
etiquetas_numericas = np.zeros(len(etiquetas))
for i, clase in enumerate(imagenes_entrenamiento.keys()):
    etiquetas_numericas[np.array(etiquetas) == clase] = i


# Crear y entrenar el clasificador de Bayes de Scikit-Learn
clasificador = GaussianNB()
clasificador.fit(np.vstack(list(vectores_rgb.values())), etiquetas_numericas)

# Nombres de las imágenes de prueba
imagenes_prueba = ['./Prueba1.jpg', './Prueba2.jpg', './Prueba3.jpg']

for imagen_prueba_nombre in imagenes_prueba:
    # Cargar la imagen de prueba y preprocesarla
    imagen_prueba = cv2.imread(imagen_prueba_nombre, cv2.IMREAD_COLOR)    
    # Corregir la inversión de colores de BGR a RGB
    imagen_prueba_rgb = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2RGB)
    
    # Aplicar el filtro gaussiano
    imagen_prueba = cv2.GaussianBlur(imagen_prueba, (0, 0), sigma)

    # Crear una imagen vacía para la clasificación
    clasificacion_imagen = np.zeros(imagen_prueba.shape[:2], dtype=np.uint8)

    # Iterar a través de cada píxel de la imagen de prueba
    for i in range(imagen_prueba_rgb.shape[0]):
        for j in range(imagen_prueba_rgb.shape[1]):
            pixel = imagen_prueba_rgb[i, j]

            # Calcular las probabilidades a posteriori para cada clase
            clase_asignada = clasificador.predict([pixel.reshape(-1)])
            
            # Asignar un valor de color diferente a cada clase
            if clase_asignada == 0:
                clasificacion_imagen[i, j] = 128
            elif clase_asignada == 1:
                clasificacion_imagen[i, j] = 64
            elif clase_asignada == 2:
                clasificacion_imagen[i, j] = 32
            else:
                clasificacion_imagen[i, j] = 0

    cv2.imwrite(f'./resultados/Imagen_Clasificada_Scikit_{imagen_prueba_nombre[2:]}', clasificacion_imagen)

cv2.destroyAllWindows()

