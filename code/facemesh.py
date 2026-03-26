#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles # Opcional: para estilos más bonitos

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# cv.namedWindow('MediaPipe FaceMesh', cv.WINDOW_NORMAL| cv.WINDOW_GUI_NORMAL)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Esto mejora la detección de ojos/labios
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

POINTS = False

print("Pulsa 'P' para alternar entre Puntos y Malla.")

for key, frame in autoStream():
    # Espejo horizontal para que sea más natural
    frame = cv.flip(frame, 1)
    h, w = frame.shape[:2]
    
    if key == ord("p"):
        POINTS = not POINTS
            
    # Mediapipe necesita RGB
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False # Mejora rendimiento ligeramente
    results = face_mesh.process(image)
    
    # Volvemos a BGR para dibujar con OpenCV
    image.flags.writeable = True
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            if POINTS:
                # Dibujar números de los puntos (modo debug)
                for p, lan in enumerate(face_landmarks.landmark):
                    # Solo dibujamos algunos puntos para no saturar, o todos si quieres
                    if p % 10 == 0: # Truco: Dibujar solo 1 de cada 10 para que sea legible
                        x = int(lan.x * w)
                        y = int(lan.y * h)
                        putText(frame, str(p), orig=(x,y), div=1, scale=0.4, color=(0,255,255))
                    
                    # Dibujar un puntito en cada landmark
                    x_all = int(lan.x * w)
                    y_all = int(lan.y * h)
                    cv.circle(frame, (x_all, y_all), 1, (0, 255, 0), -1)

            else:                
                # CORRECCIÓN AQUÍ: Usamos FACEMESH_TESSELATION en lugar de CONTOURS
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION, # <--- CAMBIO CLAVE
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                
                # Opcional: Dibujar contornos de ojos e iris (si usas refine_landmarks=True)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
    
    cv.imshow('MediaPipe FaceMesh', frame)

cv.destroyAllWindows()

