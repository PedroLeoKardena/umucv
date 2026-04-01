#!/usr/bin/env python

# script modificado para ROI interactiva y grabacion automatica (modo Alarma)

import numpy as np
import cv2 as cv
from datetime import datetime
from umucv.util import ROI, putText
from umucv.stream import autoStream

region = ROI("input")
bgsub = cv.createBackgroundSubtractorMOG2(500, 16, False)
grabando = False
video_out = None
tiempo_actual = 0
tiempo_inicio_grabacion = 0

for key, frame in autoStream():
    if region.roi:
        [x1, y1, x2, y2] = region.roi
        
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
        
        if key == ord('x'):
            region.roi = []
            
        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1, y1-8))


    fgmask = bgsub.apply(frame)
    kernel_small = np.ones((3,3), np.uint8) # Filtro leve apertura
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel_small)

    contornos, contorno = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for c in contornos:
        if cv.contourArea(c) > 150:
            x, y, w, h = cv.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            if region.roi and not grabando:
                [x1, y1, x2, y2] = region.roi
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    print("Objeto detectado. Iniciando Grabación...")
                    grabando = True
                    tiempo_inicio_grabacion = datetime.now()
                    
                    # Generar nombre identificativo en tu ordenador por fecha/hora
                    nombre_archivo = f"alerta_{tiempo_inicio_grabacion.strftime('%Y%m%d_%H%M%S')}.avi"
                    
                    # Arrancar la escritura a disco a la Tasa de 25 Frames por Segundo (FPS) y codec XVID universal (.avi)
                    H, W = frame.shape[:2]
                    fourcc = cv.VideoWriter_fourcc(*'XVID')
                    video_out = cv.VideoWriter(nombre_archivo, fourcc, 25.0, (W, H))


    #Modulo Grabador
    if grabando:
        # Añade el fotograma original actual a la película en proceso
        if video_out is not None:
            video_out.write(frame)
        
        tiempo_actual = datetime.now()
        if (tiempo_actual - tiempo_inicio_grabacion).total_seconds() > 3.0:
            #Se deja de capturar frames
            grabando = False 
            if video_out is not None:
                video_out.release() 
                video_out = None
            print(f"Clip de 3 segundos volcado. Generado en archivo: [{nombre_archivo}].")


    # 5. UI de Ayuda a Usuario
    h, w, _ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.putText(frame, 'Si quieres cerrar la aplicacion pulse: "q"', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(frame, 'Si quieres quitar la seleccion pulse: "x"', (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    # Printear la ventana con el fotograma coloreado final a tus ojos
    cv.imshow('input', frame)

# Mecanismo Safe-Shutdown: Si estas grabando a la fuerza cierras la ventana bruscamente ('Esc' o 'q'), evita el 'Video Corrupto'
if video_out is not None:
    video_out.release()

cv.destroyAllWindows()
