# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:03:40 2021

@author: Usuario
"""
import cv2 as cv

import os
import imutils

dataRuta="E:/Desktop\python/redesNeuronales/reconocimientoFacial1/Data"
listaData=os.listdir(dataRuta)
entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer.read("E:/Desktop/python/redesNeuronales/reconocimientoFacial1/entremamientoEigenFaceRecognizer.xml")
ruidos = cv.CascadeClassifier("E:/Desktop/python/redesNeuronales/reconocimientoFacial1/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml")
camara=cv.VideoCapture(0)
while True:
    respuesta,captura=camara.read()
    if respuesta==False: break
    captura=imutils.resize(captura, width=480)
    
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    cara=ruidos.detectMultiScale(grises,1.3,5)
    idcaptura=grises.copy()
    for(x,y,e1,e2)in cara:
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado,(160,160),interpolation=cv.INTER_CUBIC)
        resultado= entrenamientoEigenFaceRecognizer.predict(rostrocapturado)
        cv.putText(captura, "{}".format(resultado), (x,y-5), 1,1.3,(0,255,0),1,cv.LINE_AA)
        
        if resultado[1]<9000:
            cv.putText(captura, "{}".format(listaData[resultado[0]]), (x,y-20), 2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, y+e2),(255,0,0), 2)
            
        else:
            cv.putText(captura, "No encontrado", (x,y-20), 2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, y+e2),(255,0,0), 2)
            
        
        
    cv.imshow("resultado", captura)
    if cv.waitKey(1)==ord("s"):
        break
camara.release()
cv.destroyAllWindows()