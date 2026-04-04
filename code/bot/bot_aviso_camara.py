#!/usr/bin/env python

# script modificado para ROI interactiva y grabacion automatica (modo Alarma)

import numpy as np
import cv2 as cv
from datetime import datetime
from umucv.util import ROI, putText, check_and_download, Slider
from umucv.stream import autoStream

#Librerias para guardar imagen
from io import BytesIO
from PIL import Image

from ultralytics import YOLO
import yaml

from telegram.ext import Updater, CommandHandler
from telegram import Update, Bot

from os import environ
from dotenv import load_dotenv
load_dotenv('token.env')


MI_ID = environ['USER_ID']
UPDATER = Updater(environ['TOKEN'], use_context=True)
dispatcher = UPDATER.dispatcher
bot = UPDATER.bot

region = ROI("input")

model = YOLO("yolo11n.pt")

# class labels:
url = "https://raw.githubusercontent.com/ultralytics/ultralytics/refs/heads/main/ultralytics/cfg/datasets/coco.yaml"
check_and_download("coco.yaml", url)
labels = yaml.safe_load(open("coco.yaml", encoding="utf-8"))['names']

C = Slider("conf","input",0.5,0,1,0.01)

grabando = False
video_out = None
tiempo_actual = 0
tiempo_inicio_grabacion = 0

etiquetas_aceptadas = ["person","bicycle","car","motorcycle","bus","truck","bird","cat","dog","horse"]

img = None

for key, frame in autoStream():
    h, w, _ = frame.shape   
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    if region.roi:
        [x1, y1, x2, y2] = region.roi
        
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
        
        if key == ord('x'):
            region.roi = []
            
        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1, y1-8))


    [result] = model(rgb, verbose=False)

    for b in result.boxes:
        conf = b.conf.cpu().numpy()[0]
        if conf < C.value:
            continue

        [[x1,y1,x2,y2]] = np.array(b.xyxy.cpu()).astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv.rectangle(frame, (x1,y1), (x2, y2), color=(0,0,255))
        idx = round(b.cls.cpu().numpy()[0])
        etiqueta = labels[idx]
        putText(frame,f"{etiqueta} {conf:.2f}", (x1+4,y1+15))
        if etiqueta == "person":
            roi_persona = frame[y1:y2, x1:x2]
            #Aplicamos el difuminado
            frame[y1:y2, x1:x2] = cv.GaussianBlur(roi_persona, (51, 51), 0)
        
        if region.roi and not grabando:
            [rx1, ry1, rx2, ry2] = region.roi
            if etiqueta in etiquetas_aceptadas:
                if cx >= rx1 and cx <= rx2 and cy >= ry1 and cy <= ry2:
                    print("Objeto detectado. Iniciando Grabación...")
                    frame_arreglado = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_arreglado, mode = 'RGB')
                    byte_io = BytesIO()
                    img.save(byte_io, 'PNG')
                    byte_io.seek(0)
                    bot.send_message(MI_ID, f'Alerta: {etiqueta} detectada en la zona de interés.')
                    bot.send_photo(chat_id=MI_ID, photo=byte_io)

                    grabando = True
                    tiempo_inicio_grabacion = datetime.now()
                    nombre_archivo = f"alerta_{tiempo_inicio_grabacion.strftime('%Y%m%d_%H%M%S')}_{etiqueta}.avi"
                    H, W = frame.shape[:2]
                    fourcc = cv.VideoWriter_fourcc(*'XVID')
                    video_out = cv.VideoWriter(nombre_archivo, fourcc, 10.0, (W, H))


    #Modulo Grabador
    if grabando:
        # Añade el fotograma original actual a la película en proceso
        if video_out is not None:
            video_out.write(frame)
        
        tiempo_actual = datetime.now()
        if (tiempo_actual - tiempo_inicio_grabacion).total_seconds() > 5.0:
            #Se deja de capturar frames
            grabando = False 
            if video_out is not None:
                video_out.release() 
                video_out = None 
            print(f"Clip de 5 segundos volcado. Generado en archivo: [{nombre_archivo}].")
            


    # 5. UI de Ayuda a Usuario
    putText(frame, f'{w}x{h}')
    cv.putText(frame, 'Si quieres cerrar la aplicacion pulse: "q"', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(frame, 'Si quieres quitar la seleccion pulse: "x"', (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    # Printear la ventana con el fotograma coloreado final a tus ojos
    cv.imshow('input', frame)

# Mecanismo Safe-Shutdown: Si estas grabando a la fuerza cierras la ventana bruscamente ('Esc' o 'q'), evita el 'Video Corrupto'
if video_out is not None:
    video_out.release()

cv.destroyAllWindows()
