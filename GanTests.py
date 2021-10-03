from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from BlueScore import csvToList, BleuScoreFromTxt
from time import time


#grafico para varias listas o valores
def graph(series,names,xlabel,ylabel,title):
    indice_barras = np.arange(len(series[0]))
    ancho_barras =0.2
    for i in range(len(series)):
        plt.bar(indice_barras, series[i], width=ancho_barras, label=names[i])
        indice_barras = indice_barras + ancho_barras
    plt.legend(loc='best')
    plt.xticks(indice_barras + ancho_barras,(range(0,len(series[0]))))  
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

init= time()

referencesList=csvToList(10000,22,"corpus_good.csv")

scoreList1 = BleuScoreFromTxt(referencesList,1,'DPGAN\DPGAN_MLE_00110.txt')
scoreList2 = BleuScoreFromTxt(referencesList,1,'JSDGAN\JSDGAN_ADV_00045.txt')
scoreList3 = BleuScoreFromTxt(referencesList,1,'SeqGAN\SeqGAN_MLE_00040.txt')
end = time()

print("Time",end-init)

#Grafico detallado de puntajes 
graph([[mean(scoreList1)],[mean(scoreList2)],[mean(scoreList3)]],["DPGAN","JSDGAN","SeqGAN"],"GANs","Promedio de puntaje Bleu-4","Comparación de GANS promediada")

#Grafico de promedio de puntajes
graph([scoreList1,scoreList2,scoreList3],["DPGAN","JSDGAN","SeqGAN"],"ID del Tweet generado","Puntaje Bleu-4","Comparación entre GANS")