import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

def calcular_probabilidad_posteriori(pixel, media, covarianza):
    """
    Calcula la probabilidad a posteriori para un píxel dado los parámetros de una clase.

    Args:
        pixel (numpy.ndarray): El valor del píxel en formato RGB.
        media (numpy.ndarray): La media de los valores de color de la clase.
        covarianza (numpy.ndarray): La matriz de covarianza de los valores de color de la clase.

    Returns:
        float: La probabilidad a posteriori.
    """
    # Calcula la diferencia entre el píxel y la media de la clase
    diferencia = pixel - media
    
    # Calcula la exponencial de la expresión cuadrática en la fórmula de Bayes
    exponente = -0.5 * np.dot(np.dot(diferencia, np.linalg.inv(covarianza)), diferencia)
    
    # Calcula la probabilidad a posteriori
    probabilidad_posteriori = np.exp(exponente) / (np.sqrt(np.linalg.det(covarianza)) * (2 * np.pi) ** (3/2))
    
    return probabilidad_posteriori

# Cargar imágenes de entrenamiento para cada clase y preprocesarlas
imagenes_entrenamiento = {
    'Platano': cv2.imread('./platano.jpg', cv2.IMREAD_COLOR),
    'Huevo': cv2.imread('./huevo.jpg', cv2.IMREAD_COLOR),
    'Chile': cv2.imread('./chile.jpg', cv2.IMREAD_COLOR),
    'Fondo': cv2.imread('./fondo.jpg', cv2.IMREAD_COLOR)
}
vectores_rgb = {}

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
    vectores_rgb[f'Vector_R_{clase}'] = vector_r_sin_blancos
    vectores_rgb[f'Vector_G_{clase}'] = vector_g_sin_blancos
    vectores_rgb[f'Vector_B_{clase}'] = vector_b_sin_blancos

# Calcular media y covarianza para cada clase
medias = {}
covarianzas = {}
for clase in imagenes_entrenamiento.keys():
    media = [np.mean(vectores_rgb[f'Vector_R_{clase}']), np.mean(vectores_rgb[f'Vector_G_{clase}']), np.mean(vectores_rgb[f'Vector_B_{clase}']),]
    covarianza = np.cov([vectores_rgb[f'Vector_R_{clase}'], vectores_rgb[f'Vector_G_{clase}'], vectores_rgb[f'Vector_B_{clase}'],], rowvar=True)
    medias[clase] = media
    covarianzas[clase] = covarianza
    print(f'Media de la clase {clase}: {media}')
    print(f'Covarianza de la clase {clase}:\n{covarianza}')

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
            probabilidades_posteriori = {}
            for clase in medias.keys():
                probabilidad = calcular_probabilidad_posteriori(pixel, medias[clase], covarianzas[clase])
                probabilidades_posteriori[clase] = probabilidad

            # Asignar la clase con la probabilidad más alta al píxel
            clase_asignada = max(probabilidades_posteriori, key=probabilidades_posteriori.get)

            # Asignar un valor de color diferente a cada clase
            if clase_asignada == 'Platano':
                clasificacion_imagen[i, j] = 128
            elif clase_asignada == 'Huevo':
                clasificacion_imagen[i, j] = 64
            elif clase_asignada == 'Chile':
                clasificacion_imagen[i, j] = 32
            else:
                clasificacion_imagen[i, j] = 0

    # Guardar la imagen clasificada
    cv2.imwrite(f'./resultados/Imagen_Clasificada_{imagen_prueba_nombre[2:]}', clasificacion_imagen)

# Cerrar todas las ventanas
cv2.destroyAllWindows()


