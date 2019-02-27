#!/usr/bin/python
# Iris Detect in openCV 
# Bruno da Silva 
# email@brunodasilva.com
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import imutils

# Aqui pode ser 1 para pegar a camera local
# Nesse caso estou pegando um vídeo streaming da web
# Mas também poded ser um vídeeo local dentro das pastas
videoCapture = cv2.VideoCapture('http://192.168.100.5:8080/videofeed?.mjpg')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while True:
    (_,videoFrame) = videoCapture.read()
    # Habilitar aqui se precisar modo retrado
    #videoFrame = imutils.rotate(videoFrame,90)
    videoFrameGray = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(videoFrameGray, 1.3, 1)
    for (x, y, w, h) in faces:
        cv2.rectangle(videoFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        faceGray = videoFrameGray[y:y + h, x:x + w]
        faceColored = videoFrame[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(faceGray)
        find_iris = 0
        for (ex, ey, ew, eh) in eyes:
            eyesGray = faceGray[ey:ey + eh, ex:ex + ew]
            eyesColored = faceColored[ey:ey + eh, ex:ex + ew]
            cv2.rectangle(faceColored, (ex, ey), (ex + ew, ey + eh), (0,255, 0), 2)
            # Aqui está o truque, convertemos todo olho em blur mode/canny mode
            # Assim pessoas de olhos claros ou pouco definidos serão detectados mais facilmente
            irisBlured = cv2.GaussianBlur(eyesGray, (7,7), 1)
            irisCannyed = cv2.Canny(irisBlured,5,70,apertureSize=3)
            # Os parametrros aqui são importantes a depender da localização da camera
            # Lembre-se de aproximar perto do rosto, caso contrário precisará ajustar o radius
            irisCircles = cv2.HoughCircles(irisCannyed, cv2.cv.CV_HOUGH_GRADIENT,1,200,param1=50,param2=30,minRadius=0,maxRadius=200)
            if irisCircles is None:
                continue
            irisCircles = np.round(irisCircles[0,:]).astype("int")
            for irisCircle in irisCircles:
                # Aqui irá imprimir um circulo rosa ao redor da iris
                cv2.circle(eyesColored, (irisCircle[0], irisCircle[1]),irisCircle[2], (255,0,255), thickness=2)
                
    # Mostrar o resultado na tela        
    cv2.imshow('videoFrame', videoFrame)
    key = cv2.waitKey(1)
    if key == 27:
        break
videoCapture.release()
cv2.destroyAllWindows()
