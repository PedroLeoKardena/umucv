#!/usr/bin/env python


import cv2 as cv
import numpy as np
from umucv.stream import autoStream, sourceArgs
from umucv.util import putText
from collections import deque
import time
import math

fov_deg = 82.0 # Default FOV de mi cámara


tracks = []
track_len = 20
detect_interval = 5

corners_params = dict( maxCorners = 500,
                       qualityLevel= 0.1,
                       minDistance = 10,
                       blockSize = 7)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

prev_time = time.time()

def nothing(x): pass
cv.namedWindow('input')
cv.createTrackbar('FOV', 'input', int(fov_deg), 120, nothing)

for n, (key, frame) in enumerate(autoStream()):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    t0 = time.time()
    dt = t0 - prev_time
    prev_time = t0
    if tracks:

        # el criterio para considerar bueno un punto siguiente es que si lo proyectamos
        # hacia el pasado, vuelva muy cerca del punto incial, es decir:
        # "back-tracking for match verification between frames"
        p0 = np.float32( [t[-1] for t in tracks] )
        p1,  _, _ =  cv.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)
        p0r, _, _ =  cv.calcOpticalFlowPyrLK(gray, prevgray, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1,2).max(axis=1)
        good = d < 1

        valid_p0 = p0.reshape(-1,2)[good]
        valid_p1 = p1.reshape(-1,2)[good]
        
        direction_text = "STATIC"
        angular_velocity = 0.0
        
        if len(valid_p0) > 0:
            vx = valid_p1[:,0] - valid_p0[:,0]
            vy = valid_p1[:,1] - valid_p0[:,1]
            
            h, w = frame.shape[:2]
            cx, cy = w / 2, h / 2
            
            r0 = np.sqrt((valid_p0[:,0] - cx)**2 + (valid_p0[:,1] - cy)**2)
            r1 = np.sqrt((valid_p1[:,0] - cx)**2 + (valid_p1[:,1] - cy)**2)
            dr = r1 - r0
            
            mean_vx = np.mean(vx)
            mean_vy = np.mean(vy)
            mean_dr = np.mean(dr)
            
            movements = {
                "LEFT": mean_vx if mean_vx > 0 else 0,
                "RIGHT": -mean_vx if mean_vx < 0 else 0,
                "UP": mean_vy if mean_vy > 0 else 0,
                "DOWN": -mean_vy if mean_vy < 0 else 0,
                "FORWARD": mean_dr if mean_dr > 0 else 0,
                "BACKWARD": -mean_dr if mean_dr < 0 else 0
            }
            
            dominant_dir = max(movements, key=movements.get)
            max_val = movements[dominant_dir]
            
            if max_val > 2.0:
                direction_text = dominant_dir
                
            current_fov = cv.getTrackbarPos('FOV', 'input')
            if current_fov < 10: current_fov = 10
            
            f = (w / 2.0) / math.tan(math.radians(current_fov) / 2.0)
            
            d_px = np.sqrt(mean_vx**2 + mean_vy**2)
            angle_rad = math.atan(d_px / f)
            angle_deg = math.degrees(angle_rad)
            
            if dt > 0:
                angular_velocity = angle_deg / dt
        
        cv.putText(frame, f'Dir: {direction_text}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, f'Vel. Ang: {angular_velocity:.1f} deg/s', (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        new_tracks = []
        for t, point, ok in zip(tracks, p1.reshape(-1,2), good):
            if not ok:
                continue
            t.append( point )
            new_tracks.append(t)

        tracks = new_tracks

        cv.polylines(frame, [ np.int32(t) for t in tracks ], isClosed=False, color=(0,0,255))
        for t in tracks:
            point = np.int32(t[-1])
            cv.circle(frame, center=point, radius=2, color=(0, 0, 255), thickness=-1)

    t1 = time.time()

    if n % detect_interval == 0:
        # Creamos una máscara para indicar al detector de puntos nuevos las zona
        # permitida, que es toda la imagen, quitando círculos alrededor de los puntos
        # existentes (los últimos de las trayectorias).
        mask = np.zeros_like(gray)
        mask[:] = 255
        for x,y in [np.int32(t[-1]) for t in tracks]:
            cv.circle(mask, (x,y), 5, 0, -1)
        #cv.imshow("mask",mask)
        corners = cv.goodFeaturesToTrack(gray, mask=mask, **corners_params)
        if corners is not None:
            for [pt] in np.float32(corners):
                tracks.append( deque([pt], maxlen=track_len) )

    putText(frame, f'{len(tracks)} corners, {(t1-t0)*1000:.0f}ms' )
    cv.imshow('input', frame)
    prevgray = gray

