import cv2 as cv
from .base import MetodoClasificacion

class MetodoORB(MetodoClasificacion):
    """
    Método de detección de objetos utilizando descriptores ORB.
    Es una alternativa libre y rápida a SIFT, ideal para probar la inserción de nuevos métodos.
    """
    def __init__(self):
        self.orb = cv.ORB_create(nfeatures=500)
        # ORB usa descriptores binarios, por lo que se recomienda NORM_HAMMING
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.modelos = {}

    def precomputar_modelo(self, nombre, imagen):
        if imagen is None: return

        # Redimensionamos la imagen si es muy grande para no penalizar el rendimiento
        ancho_deseado = 640
        alto, ancho = imagen.shape[:2]

        if ancho > ancho_deseado:
            escala = ancho_deseado / ancho
            nuevo_alto = int(alto * escala)
            imagen = cv.resize(imagen, (ancho_deseado, nuevo_alto))

        kp, des = self.orb.detectAndCompute(imagen, mask=None)

        if des is not None:
            self.modelos[nombre] = (kp, des, imagen)
            print(f"[ORB] Precomputado modelo '{nombre}' con {len(kp)} puntos.")
        else:
            print(f"[ORB] No se pudo precomputar el modelo '{nombre}'.")

    def clasificar(self, frame):
        kp_frame, des_frame = self.orb.detectAndCompute(frame, mask=None)
        
        if des_frame is None or len(self.modelos) == 0:
            return frame, "Ninguno"

        mejor_nombre = "Ninguno"
        max_buenos = 0
        mejor_k0, mejor_img0, mejores_matches = None, None, []
        
        for nombre, (k0, d0, img0) in self.modelos.items():
            if d0 is None:
                continue
                
            matches = self.matcher.match(des_frame, d0)
            
            # Ordenamos los matches por distancia
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filtramos los mejores matches (distancia menor es mejor)
            good = [m for m in matches if m.distance < 50]
            
            if len(good) > max_buenos:
                max_buenos = len(good)
                mejor_nombre = nombre
                mejor_k0 = k0
                mejor_img0 = img0
                mejores_matches = good
        
        # Umbral para considerar que el objeto ha sido detectado
        if max_buenos > 15:
            # Dibujamos los primeros 20 matches para no saturar la imagen
            frame_salida = cv.drawMatches(frame, kp_frame, mejor_img0, mejor_k0, mejores_matches[:20],
                                          flags=0, matchColor=(128, 255, 128),
                                          singlePointColor=(128, 128, 128), outImg=None)
            cv.putText(frame_salida, f"Detectado: {mejor_nombre} ({max_buenos} matches)", 
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame_salida, mejor_nombre
        else:
            # Si no se detecta nada, solo mostramos los puntos de interés del frame
            cv.drawKeypoints(frame, kp_frame, frame, color=(100, 150, 255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return frame, "Ninguno"
