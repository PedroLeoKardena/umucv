#!/usr/bin/env python

# estimación de pose a partir de la tarjeta universitaria
# En esta versión añadimos una imagen fuera del plano

import cv2          as cv
import numpy        as np

from umucv.util    import read_arguments
from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose
from umucv.contours import extractContours, redu


def my_arguments(parser):
    parser.add_argument('--imagen', help='Imagen que queremos poner sobre el marker.', type=str, required=True)

args = read_arguments(my_arguments)

imvirt = cv.imread(args.imagen)

def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    # print(f)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])


stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT


K = Kfov( size, 82 ) # 82 es el campo de visión horizontal de mi cámara


#Cambiamos los valores del marker para que se correpondan con el objeto real. Los hemos puesto en cm.
#Se trata de un objeto de 8.5cm de ancho y 5.5cm de alto. El origen de coordenadas lo hemos puesto en la esquina inferior izquierda
marker = np.array(
        [[0,   0,   0],    # esquina superior-izquierda
        [8.5,   0,   0],    # esquina inferior-izquierda
        [8.5,   5.5,   0],    # esquina inferior-derecha
        [0,   5.5,   0]])   # esquina superior-derecha


def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]


for n, (key,frame) in enumerate(stream):

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    good = polygons(cs,4,3)
    poses = []
    for g in good:
        cv.polylines(frame, [g], True, (0, 255, 0), 2)
        p = bestPose(K,g,marker)
        print(p.rms)
        if p.rms < 8:
            poses += [p.M]

    for M in poses:
        
        # las coordenadas de sus 4 esquinas
        # (se pueden sacar del bucle de captura)
        h,w = imvirt.shape[:2]
        src = np.array([[0,0],[0,h],[w,h],[w,0]])
        
        # decidimos dónde queremos poner esas esquinas en el sistema de referencia del marcador
        # (si no cambian se puede sacar del bucle de captura)
        world = np.array([[6.4, 2.9,0],[6.4,5.1,0],[8.1,5.1,0],[8.1,2.9,0]])
        
        # calculamos dónde se proyectarán en la imagen esas esquinas
        # usamos la matriz de cámara estimada
        dst = htrans(M, world)

        # calculamos la transformación
        #H, _ = cv.findHomography(src,dst)
        # igual que findHomography pero solo con 4 correspondencias
        H = cv.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        # la aplicamos encima de la imagen de cámara
        cv.warpPerspective(imvirt,H,size,frame,0,cv.BORDER_TRANSPARENT)


    cv.imshow('source',frame)
    
