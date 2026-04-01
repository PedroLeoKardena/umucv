#!/usr/bin/env python

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from collections import deque
import math

def nothing(x): pass

cv.namedWindow('medidor')

cv.createTrackbar('fov', 'medidor', 46, 80, nothing)
cv.createTrackbar('Z', 'medidor', 22, 290, nothing)
cv.createTrackbar('A', 'medidor', 8, 20, nothing)
cv.createTrackbar('X', 'medidor', 50, 100, nothing) 

points = deque(maxlen=2)

def manejador(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

cv.setMouseCallback('medidor', manejador)

cv.imshow('medidor', np.zeros((100, 100, 3), dtype=np.uint8))
cv.waitKey(100)

for key, frame in autoStream():
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    
    display = frame.copy()
    
    try:
        fov_tb = cv.getTrackbarPos('fov', 'medidor')
        Z_tb = cv.getTrackbarPos('Z', 'medidor')
        A_tb = cv.getTrackbarPos('A', 'medidor')
        X_tb = cv.getTrackbarPos('X', 'medidor')
    except:
        # Prevención de fallos
        fov_tb, Z_tb, A_tb, X_tb = 46, 22, 8, 50
    
    # Conversiones al espacio real
    fov_deg = fov_tb + 10
    if fov_deg <= 0.1: fov_deg = 0.1
    
    f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    
    Z = Z_tb / 10.0 + 1.0       # De 1m a 30m
    alt = A_tb / 10.0           # Altura sobre nivel 0 (Suelo)
    X_offset = (X_tb - 50)/10.0 # Desplazamiento lateral de -5m a 5m
    
    cv.line(display, (0, int(cy)), (w, int(cy)), (150, 150, 150), 1)
    cv.line(display, (int(cx)-20, int(cy)), (int(cx)+20, int(cy)), (150, 150, 150), 1)
    cv.line(display, (int(cx), int(cy)-20), (int(cx), int(cy)+20), (150, 150, 150), 1)
    
    v_base = int(cy + f * alt / Z)
    
    if v_base < h and Z > 0:
        cv.line(display, (0, v_base), (w, v_base), (255, 255, 255), 3)
    
    if Z > 0:
        for X_wall in range(-30, 31):
            X_real = X_wall + X_offset
            u = int(cx + f * X_real / Z)
            
            cv.line(display, (u, 0), (u, v_base), (200, 200, 200), 1)

            #Estos nos permite dibujar las lineas desde las verticales hacia la imagen de la camara             
            u_cerca = int(cx + f * X_real / 0.1)
            v_cerca = int(cy + f * alt / 0.1)
            cv.line(display, (u, v_base), (u_cerca, v_cerca), (200, 200, 200), 1)

        for H in range(1, 15):
            Y_cam = alt - H
            v = int(cy + f * Y_cam / Z)
            if 0 <= v < h:
                cv.line(display, (0, v), (w, v), (200, 200, 200), 1)
                cv.putText(display, str(H), (int(cx)+5, v-5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv.LINE_AA)

        for z_floor in np.arange(1, math.ceil(Z), 1):
            if z_floor <= 0: continue
            v_floor = int(cy + f * alt / z_floor)
            if v_base < v_floor < h:
                cv.line(display, (0, v_floor), (w, v_floor), (200, 200, 200), 1)
                
        u_0 = int(cx + f * X_offset / Z)
        cv.putText(display, "0", (u_0 + 5, v_base - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    # Motor Interactivo de Mediciones Reales Segmentadas
    for p in points:
        cv.circle(display, p, 3, (0, 0, 255), -1)
    
    if len(points) == 2:
        cv.line(display, points[0], points[1], (0, 0, 255), 1)
        
        u1, v1 = points[0]
        u2, v2 = points[1]
        
        # Ray casting hasta plano Z definido en la interacciÃ³n del usuario
        x1_real = (u1 - cx) * Z / f
        y1_real = (v1 - cy) * Z / f
        x2_real = (u2 - cx) * Z / f
        y2_real = (v2 - cy) * Z / f
        
        dist_m = math.dist((x1_real, y1_real), (x2_real, y2_real))
        mx, my = (u1 + u2) // 2, (v1 + v2) // 2
        
        # Formateo como cm para resoluciones del submúltiplo
        texto_dist = f"{dist_m*100:.0f} cm" if dist_m < 1 else f"{dist_m:.2f} m"
        cv.putText(display, texto_dist, (mx, my - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
        
    # Superposición HUD de variables globales (Arriba Izquierda)
    cv.putText(display, f"FOV={fov_deg:.1f} deg, f={int(f)}px ({w}x{h})", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(display, f"Z={Z:.1f} m", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(display, f"alt={alt:.1f} m", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    
    cv.imshow('medidor', display)