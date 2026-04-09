import cv2 as cv
import os
import sys
import glob

from umucv.stream import autoStream
from umucv.util import read_arguments
from metodos import sift, hands, embedder

METODOS_DISPONIBLES = {
    "SIFT": sift.MetodoSIFT,
    "MEDIAPIPE": embedder.MetodoEmbedder,
}

def my_arguments(parser):
    parser.add_argument('--models', help='carpeta con el conjunto de imágenes a reconocer (obligatorio)', type=str, required=True)
    parser.add_argument('--method', help='nombre del método de comparación (obligatorio)', type=str, required=True)

def iniciar_camara(clasificador):
    print("\n[INFO] Abriendo cámara... Pulsa 'q' en la ventana para salir al menú.")
    for key, frame in autoStream():
        frame, etiqueta = clasificador.clasificar(frame)
        cv.imshow("Clasificador", frame)

    cv.destroyAllWindows()


if __name__ == "__main__":

    args = read_arguments(my_arguments)
    metodo_elegido = args.method.upper()

    if not os.path.isdir(args.models):
        print(f"Error: La ruta '{args.models}' no existe o no es una carpeta.")
        sys.exit(1)

    imagenes_paths = glob.glob(os.path.join(args.models, '*.png')) + \
                    glob.glob(os.path.join(args.models, '*.jpg')) + \
                    glob.glob(os.path.join(args.models, '*.jpeg'))

    if not imagenes_paths:
        print(f"Aviso: No se han encontrado imágenes (.jpg, .jpeg, .png) en '{args.models}'.")
    else:
        print(f"Se han encontrado {len(imagenes_paths)} imágenes en la carpeta '{args.models}'.")

    while True:
        print("Introduzca opción:")
        print(f"\t1. Reconocimiento de objetos con método introducido: {args.method}")
        print(f"\t2. Reconocedor de gestos de manos.")
        print(f"\t3. Añadir nuevo método. Esto requerirá reiniciar la aplicación.")
        print(f"\t4. Salir")

        opcion = input("Opción: ")
        if opcion == "1":
            if metodo_elegido not in METODOS_DISPONIBLES:
                print(f"Error: El método '{args.method}' no existe.")
                sys.exit(1)
            
            print("Reconociendo objetos con método introducido: ", args.method)
            clasificador = METODOS_DISPONIBLES[metodo_elegido]()

            print("Precomputando modelos...")
            for ruta in imagenes_paths:
                nombre = os.path.basename(ruta)
                etiqueta = os.path.splitext(nombre)[0]
                img = cv.imread(ruta)
                clasificador.precomputar_modelo(etiqueta, img)

            iniciar_camara(clasificador)  
        elif opcion == "2":

            print("Reconociendo gestos de manos")
            clasificador = hands.MetodoManos()

            print("Precomputando modelos...")
            for ruta in imagenes_paths:
                nombre = os.path.basename(ruta)
                etiqueta = os.path.splitext(nombre)[0]
                img = cv.imread(ruta)
                clasificador.precomputar_modelo(etiqueta, img)

            iniciar_camara(clasificador)  

        elif opcion == "3":
            print("\n[INFO] Para añadir un método nuevo:")
            print("1. Crea un archivo 'nuevo_metodo.py' en la carpeta 'metodos/'.")
            print("2. Haz que herede de 'MetodoClasificacion'.")
            print("3. Añádelo al diccionario METODOS_DISPONIBLES en 'app.py'.")
            print("4. Reinicia la aplicación.")
        elif opcion == "4":
            print("Saliendo")
            sys.exit(0)
        else:
            print("Opción no válida")
    