import matplotlib.pyplot as plt
import cv2
from skimage import io, color
from PIL import Image
import numpy as np
import os
import pydicom
import imghdr


# Ejercicio 1: Leer y desplegar imágenes con diferentes paquetes
image_path = "p0/resources/lena_color_512.tif"

# Cargar y mostrar la imagen con Matplotlib
image_matplotlib = plt.imread(image_path)
plt.figure(figsize=(5, 5))
plt.imshow(image_matplotlib)
plt.title("Matplotlib Image")
plt.show()

# Cargar y mostrar la imagen con OpenCV
image_opencv = cv2.imread(image_path)
cv2.imshow("OpenCV Image", image_opencv)
cv2.waitKey(2000)  # Espera 2 segundos (2000 milisegundos)
cv2.destroyAllWindows()

# Cargar y mostrar la imagen con Scikit-Image
image_skimage = io.imread(image_path)
plt.figure(figsize=(5, 5))
plt.imshow(image_skimage)
plt.title("Scikit-Image Image")
plt.show()

# Cargar y mostrar la imagen con PIL
image_pil = Image.open(image_path)
image_pil.show()





# Ejercicio 2: Imprimir el tipo de imagen, el tamaño y el tipo de dato
# Obtener el tipo de archivo de imagen
image_type = imghdr.what(image_path)
if image_type is not None:
    print("Tipo de imagen:", image_type)
else:
    print("No se pudo determinar el tipo de imagen.")
print("Tipo de representacion con Matplotlib:", type(image_matplotlib))
print("Tamaño de imagen con Matplotlib:", image_matplotlib.shape)
print("Tipo de dato con Matplotlib:", image_matplotlib.dtype)





# Ejercicio 3: Cambiar espacio de color con OpenCV y Scikit-Image
# Cargar y mostrar la imagen original con Matplotlib
image_matplotlib = plt.imread(image_path)
plt.figure(figsize=(5, 5))
plt.imshow(image_matplotlib)
plt.title("Original Image")
plt.show()

# 3.1 RGB a Escala de grises con OpenCV
gray_image_opencv = cv2.cvtColor(image_matplotlib, cv2.COLOR_RGB2GRAY)
plt.figure(figsize=(5, 5))
plt.imshow(gray_image_opencv, cmap="gray")
plt.title("Gray Image (OpenCV)")
plt.show()

# 3.2 RGB a YUV con OpenCV
yuv_image_opencv = cv2.cvtColor(image_matplotlib, cv2.COLOR_RGB2YUV)
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(yuv_image_opencv, cv2.COLOR_YUV2RGB))
plt.title("YUV Image (OpenCV)")
plt.show()

# 3.3 RGB a HSV con Scikit-Image
hsv_image_skimage = color.rgb2hsv(image_matplotlib)
plt.figure(figsize=(5, 5))
plt.imshow(color.hsv2rgb(hsv_image_skimage))
plt.title("HSV Image (Scikit-Image)")
plt.show()






# Ejercicio 4: Despliega la paleta de colores de RGB por separado.
image_matplotlib = plt.imread(image_path)
plt.figure(figsize=(5, 5))
plt.imshow(image_matplotlib)
plt.title("Original Image")
plt.colorbar()
plt.show()





# Ejercicio 5: De una imagen que usted escoja, dejarla en escala de grises y procure que sea igual en renglones y en
# columnas. Programe una función que realice decimación de una imagen, reduciendola a la mitad de su tamaño original.
# Y promediando en grupos de 4 pixeles. Pruebe con su imagen.
image = cv2.imread(image_path)
# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Función para realizar la decimación y promedio
def decimate_and_average(image):
    # Obtener las dimensiones originales de la imagen
    height, width = image.shape[:2]
    # Decimar la imagen reduciéndola a la mitad de su tamaño original
    decimated_image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
    # Inicializar una imagen para el resultado final
    averaged_image = np.zeros_like(decimated_image)
    # Realizar el promedio en grupos de 4 píxeles
    for i in range(0, height // 2, 2):
        for j in range(0, width // 2, 2):
            # Obtener los 4 píxeles en el grupo
            group = decimated_image[i:i + 2, j:j + 2]
            # Calcular el promedio de los píxeles en el grupo
            average_color = np.mean(group)
            # Establecer el promedio como el color del grupo
            averaged_image[i:i + 2, j:j + 2] = average_color
    return averaged_image
# Aplicar la decimación y promedio
result_image = decimate_and_average(gray_image)
# Mostrar la imagen original y la imagen resultante
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Imagen Original')
plt.subplot(1, 2, 2)
plt.imshow(result_image, cmap='gray')
plt.title('Imagen Decimada y Promediada')
plt.show()





# Ejercicio 6: Convertir la imagen peppers_color.tif a escala de grises y recortar
peppers_image = cv2.imread("p0/resources/peppers_color.tif")
peppers_gray = cv2.cvtColor(peppers_image, cv2.COLOR_BGR2GRAY)
x, y, w, h = 100, 100, 200, 200
peppers_cropped = peppers_gray[y : y + h, x : x + w]
cv2.imwrite("p0/resources/peppers_cropped.jpg", peppers_cropped)





# Ejercicio 7: Leer y desplegar imagen RAW
width, height = 800, 600
raw_image_data = np.fromfile("p0/resources/rosa800x600.raw", dtype=np.uint8)
raw_image = np.reshape(raw_image_data, (height, width))
cv2.imshow("RAW Image", raw_image)
cv2.waitKey(2000)  # Espera 2 segundos (2000 milisegundos)
cv2.destroyAllWindows()





# Ejercicio 8: Lectura de Video archivo .avi (asegúrate de tener OpenCV instalado)
cap = cv2.VideoCapture("p0/resources/0X2A8498756C4D6E82_corto.avi")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Video", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()





# Ejercicio 9: Imagen DICOM
dicom_folder = "p0/resources/DICOM"
dicom_files = [
    os.path.join(dicom_folder, filename)
    for filename in os.listdir(dicom_folder)
    if filename.endswith(".dcm")
]
dicom_files.sort()

for dicom_file in dicom_files:
    ds = pydicom.dcmread(dicom_file)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.title(f"Frame: {ds.InstanceNumber}")
    plt.pause(0.1)
    plt.clf()

plt.show()
