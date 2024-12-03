import cv2
import numpy as np
#cargamos imagen Goku
image = cv2.imread('static/goku.jpg')
new_height, new_width = 400, 500
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#obtener dimensiones de la imagen Naruto
image = cv2.imread('static/naruto.jpg')
hight, width = image.shape[:2]
center = (width/2, hight/2)
new_height, new_width = 400, 500
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#rotar la imagen Harry Potter
image = cv2.imread('static/harry_potter.jpg')
angulo = 90
matrix = cv2.getRotationMatrix2D(center, angulo, 1.0)
rotated = cv2.warpAffine(image, matrix, (width, hight))
new_height, new_width = 400, 500
cv2.imshow('Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Definir la matriz de traslación Sailor Moon
image = cv2.imread('static/Sailor_Moon.jpg')
tx, ty = 100, 50 
M = np.float32([[1, 0, tx], [0, 1, ty]])
#Aplicar la matriz de traslación a la imagen
translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
#Mostratre la imagen trasladada
new_height, new_width = 400, 500
cv2.imshow('Image', translated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Escala
#Definir la nueva altura y ancho Ranma 1_2
image = cv2.imread('static/Ranma1_2.jpg')
#Aplicar la esca de la imagen
scaled = cv2.resize(image, (new_width, new_height))
#Mostrar la imagen escalada
new_height, new_width = 400, 500
cv2.imshow('Image', scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Recorte
#Definir las coordenas del área de interes ROI Pokemon
image = cv2.imread('static/Pokemon.jpeg')
x, y, w, h = 100, 100, 200, 150
#Recortar la región de interés ROI
cropped = image[y:y+h, x:x+w]
#Mostrar la imagen recortada
new_height, new_width = 400, 500
cv2.imshow('Image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Suavizado
# Aplicar el filtro Gaussiano para suavisar la imagen
smoothed = cv2.GaussianBlur(image, (5, 5), 0)
# Mostrar la imagen suavizada
new_height, new_width = 400, 500
cv2.imshow('Image', smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Realce
# Definir el kernel para el filtro de afilado
kernel = np.array([[-1, -1, -1], 
                   [-1, 9, -1], 
                   [-1, -1, -1]])
# Aplicar el filtro de afilado para realzar los detalles
sharpened = cv2.filter2D(image, -1, kernel)
# Mostrar la imagen realzada
new_height, new_width = 400, 500
cv2.imshow('Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Detección de bordes
# Cargar la imagen en escala de grises supercampeones
image = cv2.imread('static/SuperCampeones.jpg', cv2.IMREAD_GRAYSCALE)
# Aplicar el operador Sobel para detectar bordes
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
# Combinar las respuestas en magnitud 
edges = cv2.magnitude(sobel_x, sobel_y)
# Normalizar los valores para mostrar la imagen correctamnete
edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# Mostrar la imagen
new_height, new_width = 400, 500
cv2.imshow('Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
