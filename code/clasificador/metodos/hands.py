import cv2 as cv
import mediapipe as mp
import numpy as np

from scipy.spatial import procrustes
from .base import MetodoClasificacion

#Funcion para extraer los puntos (x,y) para cada valor de la mano
def extraer_puntos(hand_landmarks):
    puntos = []
    for landmark in hand_landmarks.landmark:
        puntos.append([landmark.x, landmark.y])
    
    return np.array(puntos)

class MetodoManos(MetodoClasificacion):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
            
        self.modelos_gestos = {} 

    

    def precomputar_modelo(self, nombre, imagen):
        """VERSIÓN DEPURADO"""
        if imagen is None: return
        #imagen = cv.flip(imagen, 1)
        image_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            self.modelos_gestos[nombre] = extraer_puntos(results.multi_hand_landmarks[0])
            print(f"[Manos] Precomputado gesto '{nombre}'.")
            
            img_debug = imagen.copy()
            
            self.mp_drawing.draw_landmarks(
                img_debug,
                results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            nombre_ventana = f"Comprobando modelo: {nombre}"
            cv.imshow(nombre_ventana, img_debug)
            print(">>> Pulsa la tecla ESPACIO para cargar el siguiente modelo...")
            
            while True:
                tecla = cv.waitKey(0) & 0xFF
                if tecla == 32: # 32 es el código ASCII del espacio
                    break
                    
            cv.destroyWindow(nombre_ventana)

        else:
            print(f"[AVISO] MediaPipe no ha logrado detectar ninguna mano en la imagen '{nombre}'.")    

        """
        if imagen is None: return
        image_rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            self.modelos_gestos[nombre] = extraer_puntos(results.multi_hand_landmarks[0])
            print(f"[Manos] Precomputado gesto '{nombre}'.")
        """
        
    
    def clasificar(self, frame):
        if frame is None:
            return frame, "Desconocido"

        frame = cv.flip(frame, 1);
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        etiqueta_detectada = "Desconocido"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
                puntos_camara = extraer_puntos(hand_landmarks)
                mejor_distancia = float('inf')

                for etiqueta_modelo, puntos_modelo in self.modelos_gestos.items():
                    _, _, disparidad = procrustes(puntos_modelo, puntos_camara)

                    if disparidad < mejor_distancia:
                        mejor_distancia = disparidad
                        etiqueta_detectada = etiqueta_modelo

                umbral_aceptacion = 0.13
                if mejor_distancia < umbral_aceptacion:
                    # Pintamos la etiqueta ganadora en la pantalla
                    cv.putText(frame, f"Gesto: {etiqueta_detectada.upper()}. Distancia: {mejor_distancia:.2f}", (20, 50), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    etiqueta_detectada = "Desconocido"
                    cv.putText(frame, "Gesto no reconocido", (20, 50), 
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame, etiqueta_detectada