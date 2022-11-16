# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:03:40 2021

@author: Usuario
"""
import cv2 as cv
import os
import numpy as np
from time import time
dataRuta="E:/Desktop/python/redesNeuronales/reconocimientoFacial1/Data"
listaData=os.listdir(dataRuta)
#print ("data", listaData)
ids=[]
rostrosData=[]
id=0
tiempoInicial=time()
for fila in listaData:
    rutacompleta=dataRuta+"/"+ fila
    print("Inica lectura...")
    for archivo in os.listdir(rutacompleta):
        
        print("Imagenes: ",fila +"/"+archivo)
        ids.append(id)
        rostrosData.append(cv.imread(rutacompleta+ "/"+archivo,0))
        

        #imagenes=cv.imread(rutacompleta+"/"+archivo,0)
    id =id+1
    tiempoFinalLectura=time()
    tiempoTotalLectura=tiempoFinalLectura-tiempoInicial
    print("Tiempo Total lectura: ",tiempoTotalLectura)

entrenamientoEigenFaceRecognizer=cv.face.EigenFaceRecognizer_create()
print("Iniciando el entrenamiento.... Espere...")
entrenamientoEigenFaceRecognizer.train(rostrosData,np.array(ids))
TiempoFinalEntrenamiento=time()
tiempoTotalEntrenamiento=TiempoFinalEntrenamiento-tiempoTotalLectura
print("Tiempo entrenamiento Total: ",tiempoTotalEntrenamiento)

entrenamientoEigenFaceRecognizer.write("E:/Desktop/python/redesNeuronales/reconocimientoFacial1/entremamientoEigenFaceRecognizer.xml")
print("Entrenamiento terminado")


