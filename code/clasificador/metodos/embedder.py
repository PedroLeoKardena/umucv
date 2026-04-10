import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from umucv.util import check_and_download
from .base import MetodoClasificacion

class MetodoEmbedder(MetodoClasificacion):
    def __init__(self):
        check_and_download("embedder.tflite","https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite")
        options = vision.ImageEmbedderOptions(
            base_options=python.BaseOptions(model_asset_path='embedder.tflite'),
            l2_normalize=True, quantize=True)
        self.embedder = vision.ImageEmbedder.create_from_options(options)
        
        self.modelos = {}

    def precomputar_modelo(self, nombre, imagen):
        if imagen is None: return

        ancho_deseado = 640
        alto, ancho = imagen.shape[:2]

        if ancho > ancho_deseado:
            escala = ancho_deseado / ancho
            nuevo_alto = int(alto * escala)
            imagen = cv.resize(imagen, (ancho_deseado, nuevo_alto))

        mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(imagen, cv.COLOR_BGR2RGB))
        res = self.embedder.embed(mpimage)
        if res.embeddings:
            self.modelos[nombre] = res.embeddings[0]
            print(f"[Embedder] Precomputado modelo '{nombre}'.")

    def clasificar(self, frame):
        mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        res = self.embedder.embed(mpimage)
        
        if not res.embeddings or len(self.modelos) == 0:
            return frame, "Ninguno"
            
        descriptor_frame = res.embeddings[0]
        
        mejor_nombre = "Ninguno"
        max_similitud = 0.0

        for nombre, descriptor_modelo in self.modelos.items():
            similitud = vision.ImageEmbedder.cosine_similarity(descriptor_frame, descriptor_modelo)
            if similitud > max_similitud:
                max_similitud = similitud
                mejor_nombre = nombre

        #PARA DEPURAR:
        #print(f"Mejor coincidencia: {mejor_nombre} -> Similitud: {max_similitud:.3f}")
        
        W = frame.shape[1]

        umbral = 0.20
        if max_similitud > umbral: 
            cv.rectangle(frame, (0,0), (int(max_similitud*W), 20), color=(0,255,0), thickness=-1)
            cv.putText(frame, f"{mejor_nombre} ({max_similitud:.2f})", (10, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame, mejor_nombre
            
        return frame, "Ninguno"