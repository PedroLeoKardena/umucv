#!/usr/bin/env python

from ultralytics import YOLO
import cv2 as cv
from umucv.stream import autoStream
import glob
import os

rutas = glob.glob("runs/detect/train*/weights/best.pt")
if not rutas:
    print("Error: No se ha encontrado ningún modelo entrenado.")
    exit()

ruta_modelo = max(rutas, key=os.path.getmtime)
print(f"Cargando modelo: {ruta_modelo}")

model = YOLO(ruta_modelo)

print("Pulsa la 'q' en la ventana para salir.")
for key, frame in autoStream():
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    [result] = model(rgb, verbose=False)
    
    frame_pintado = cv.cvtColor(result.plot(), cv.COLOR_RGB2BGR)
    
    cv.imshow("ML Personalizado", frame_pintado)
