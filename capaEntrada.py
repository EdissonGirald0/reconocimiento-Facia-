import cv2 as cv
import numpy as np
import os
import imutils
modelo= "FotosLuciana"
ruta="E:/Desktop/python/redesNeuronales/reconocimientoFacial1/Data"
rutaCompleta=ruta +"/"+ modelo
if not os.path.exists(rutaCompleta):
    os.makedirs(rutaCompleta)
id=350
camara=cv.VideoCapture(0)
ruidos = cv.CascadeClassifier("E:/Desktop/python/redesNeuronales/reconocimientoFacial1/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml")

while True:
    respuesta,captura=camara.read()
    if respuesta==False:break
    captura=imutils.resize(captura,width=640)

    grises=cv.cvtColor(captura,cv.COLOR_BGR2GRAY)
    idCaptura=captura.copy()

    cara=ruidos.detectMultiScale(grises,1.3,5)

    for(x,y,e1,e2)in cara:
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)
        rostrocapturado=idCaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado,(160,160),interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutaCompleta+"/imagen_{}.jpg".format(id),rostrocapturado)
        id=id+1
    cv.imshow("Resultado Rostro", captura)
    if id==550:
        break
camara.release()
cv.destroyAllWindows()



