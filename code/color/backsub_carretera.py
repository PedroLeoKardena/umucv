#!/usr/bin/env python

# https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import numpy as np
from datetime import datetime
from umucv.stream import autoStream

bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)

inicio = datetime.now()

contador = 0
vehiculos_previos = []

for key,frame in autoStream():
    fgmask = bgsub.apply(frame)

    #Kernel que nos permite eliminar pequeños puntos blancos de ruido
    kernel_small = np.ones((3,3), np.uint8)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel_small)
    
    #Kernel que nos permite rellenar los huecos negros "DENTRO" de los coches para no partirlos en trozos
    kernel_large = np.ones((7,7), np.uint8)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel_large)

    # Obtenemos la resolución dinámicamente a partir del 'frame'
    H, W = frame.shape[:2]
    
    # La máscara (fgmask) es 1 canal (blanco y negro).
    mask_color = cv.cvtColor(fgmask, cv.COLOR_GRAY2BGR)

    #Encontramos los contornos
    contornos, contorno = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    vehiculos_actuales = []
    color_linea = (0, 0, 255) # Por defecto mantenemos la línea roja
    
    #Por cada contorno de un area mayor que cierta cantidad de pixeles, los procesamos
    for c in contornos:
        if cv.contourArea(c) > 200:
            cv.drawContours(mask_color, c, -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            
            # Dibujamos en los frames el centroide
            cv.circle(mask_color, (cx, cy), 5, (0, 0, 255), -1)
            
            for px, py in vehiculos_previos:
                distancia = np.hypot(cx - px, cy - py)
                if distancia < 50:
                    #Comprobamos si el coche acaba de cruzar la mitad de la pantalla (W//2)
                    if (px < W // 2 and cx >= W // 2) or (px > W // 2 and cx <= W // 2):
                        contador += 1
                        color_linea = (0, 255, 0) #Parpadeo verde al pasar coche sumado
                    break
            
            vehiculos_actuales.append((cx, cy))
            
    # Guardamos el estado actual para que sea evaluado en el próximo 'frame' temporal
    vehiculos_previos = vehiculos_actuales
    
    # Dibujamos una línea vertical en el medio del ancho (W // 2) con su color dinámico
    cv.line(mask_color, (W // 2, H // 4), (W // 2, (H - H//4)), color_linea, 2)

    cv.putText(mask_color, f"Si quieres cerrar la aplicacion pulse: \"q\"", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(frame, f"Si quieres cerrar la aplicacion pulse: \"q\"", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    
    # Informacion del total contado
    cv.imshow('original',frame)
    cv.imshow('mask', mask_color)

    

cv.destroyAllWindows()

fin = datetime.now()
tiempo_total = fin - inicio
print(f"Hora de inicio capturada = {inicio}")
print(f"Hora de fin capturada = {fin}")
print(f"Tiempo total: {tiempo_total.total_seconds():.2f} segundos")
print(f"Coches totales detectados: {contador}")
print(f"Coches por segundo: {contador/tiempo_total.total_seconds():.2f}")
