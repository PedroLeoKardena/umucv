import os
import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util import putText, read_arguments
from collections import deque
import sys


os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"

def my_arguments(parser):
    parser.add_argument('--refs', help='Archivo de texto con las referencias. Cada línea debe tener: u v X Y. (u,v) = pixeles (ancho, alto); (X,Y) = medidas reales en centímetros', type=str, required=True)

args = read_arguments(my_arguments)

# Leemos las referencias (fichero en .txt)
try:
    refs = np.loadtxt(args.refs)
except Exception as e:
    print(f"Error al leer el archivo de referencias: {e}")
    sys.exit(1)

if len(refs) < 4:
    print("Error: Se necesitan al menos 4 puntos de referencia.")
    sys.exit(1)

puntos_pixeles = refs[:, :2].astype(np.float32)
puntos_reales = refs[:, 2:].astype(np.float32)

# Calcular la homografía de rectificación
H, _ = cv.findHomography(puntos_pixeles, puntos_reales)

if H is None:
    print("Error: No se ha podido calcular la homografía con los puntos dados.")
    sys.exit(1)

# Marcamos los dos puntos para calcular su distancia real
points = deque(maxlen=2)

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

cv.namedWindow("Deformacion")
cv.setMouseCallback("Deformacion", fun)

print("\n" + "="*50)
print(" INSTRUCCIONES:")
print(" Haz clic en 2 puntos en la ventana 'Deformacion' para medir la distancia real entre ellos.")
print(" Pulsa 'q' en la ventana para salir.")
print("="*50 + "\n")

# Calcular una escala visual dinámica para que la imagen rectificada siempre tenga un ancho manejable (ej. ~600px)
max_real_x = np.max(puntos_reales[:, 0])
if max_real_x > 0:
    escala_vis = 600.0 / max_real_x
else:
    escala_vis = 30.0

S = np.array([[escala_vis, 0, 0],
              [0, escala_vis, 0],
              [0, 0, 1]])
H_vis = S @ H

# Calculamos un tamaño de ventana decente para la vista rectificada
max_x = int(np.max(puntos_reales[:, 0]) * escala_vis) + 50
max_y = int(np.max(puntos_reales[:, 1]) * escala_vis) + 50

for key, frame in autoStream():
    display = frame.copy()
    
    # Dibujar puntos de referencia originales en amarillo para tenerlos de guía
    for pt in puntos_pixeles:
        cv.circle(display, tuple(pt.astype(int)), 3, (0, 255, 255), -1)
    
    #Dibujar el rectángulo que marca la zona de referencia 
    if len(puntos_pixeles) == 4:
        cv.polylines(display, [puntos_pixeles.astype(int)], isClosed=True, color=(0, 255, 255), thickness=1)

    # Dibujar los puntos que marca el usuario
    for p in points:
        cv.circle(display, p, 4, (0, 0, 255), -1)
        
    if len(points) == 2:
        cv.line(display, points[0], points[1], (0, 0, 255), 2)
        
        # Transformamos puntos
        # perspectiveTransform requiere un array de forma (1, N, 2)
        pts_pix = np.array([[points[0], points[1]]], dtype=np.float32)
        pts_real = cv.perspectiveTransform(pts_pix, H)
        
        real1 = pts_real[0][0]
        real2 = pts_real[0][1]
        
        # Calcular y Dibujar la distancia euclídea
        d = np.linalg.norm(real2 - real1)
        
        c = np.mean(points, axis=0).astype(int)
        putText(display, f'{d:.2f} cm', c)

    cv.imshow('Deformacion', display)
    
    # Mostrar la imagen rectificada para "comprobar las operaciones"
    rectificada = cv.warpPerspective(frame, H_vis, (max_x, max_y))
    cv.imshow('Rectificada', rectificada)

cv.destroyAllWindows()
