import cv2 as cv
from .base import MetodoClasificacion

class MetodoSIFT(MetodoClasificacion):
    #Funcion de iniciación
    def __init__(self):
        self.sift = cv.SIFT_create(nfeatures=500)
        self.matcher = cv.BFMatcher()
        self.modelos = {}
    
    def precomputar_modelo(self, nombre, imagen):
        if imagen is None: return
        kp, des = self.sift.detectAndCompute(imagen, mask=None)
        if des is not None:
            self.modelos[nombre] = (kp, des, imagen)
            print(f"[SIFT] Precomputado modelo '{nombre}' con {len(kp)} puntos.")

    def clasificar(self, frame):
        #Codigo extraido de sift.py (carpeta SIFT)
        kp_frame, des_frame = self.sift.detectAndCompute(frame, mask=None)
        
        if des_frame is None or len(self.modelos) == 0:
            return frame, "Ninguno"

        mejor_nombre = "Ninguno"
        max_buenos = 0
        mejor_k0, mejor_img0, mejores_matches = None, None, []
        
        for nombre, (k0, d0, img0) in self.modelos.items():
            matches = self.matcher.knnMatch(des_frame, d0, k=2)
            good = []
            for m in matches:
                if len(m) >= 2:
                    best, second = m
                    if best.distance < 0.75 * second.distance:
                        good.append(best)
            
            if len(good) > max_buenos:
                max_buenos = len(good)
                mejor_nombre = nombre
                mejor_k0 = k0
                mejor_img0 = img0
                mejores_matches = good
        
        if max_buenos > 10:
            frame_salida = cv.drawMatches(frame, kp_frame, mejor_img0, mejor_k0, mejores_matches,
                                          flags=0, matchColor=(128,255,128),
                                          singlePointColor=(128,128,128), outImg=None)
            cv.putText(frame_salida, f"Detectado: {mejor_nombre} ({max_buenos} matches)", 
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame_salida, mejor_nombre
        else:
            cv.drawKeypoints(frame, kp_frame, frame, color=(100,150,255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return frame, "Ninguno"
