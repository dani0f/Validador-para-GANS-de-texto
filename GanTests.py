from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from BlueScore import csvToList, BleuScoreFromTxt


#grafico de solo una lista de score
def graphType1(scoreList,title):
    eje_x = range(0,len(scoreList))
    eje_y = scoreList
    plt.bar(eje_x, eje_y)
    plt.ylabel('Puntaje')
    plt.xlabel('Textos generados')
    plt.title(title)
    plt.show()   

#grafico para varias listas o valores
def graphType2(series,names,xlabel,ylabel,title):
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



referencesList=csvToList(100,22,"corpus_good.csv")

scoreList1 = BleuScoreFromTxt(referencesList,100,1,'DPGAN\DPGAN_MLE_00110.txt')
scoreList2 = BleuScoreFromTxt(referencesList,100,1,'JSDGAN\JSDGAN_ADV_00045.txt')
scoreList3 = BleuScoreFromTxt(referencesList,100,1,'SeqGAN\SeqGAN_MLE_00040.txt')

#Grafico detallado de puntajes 
graphType2([[mean(scoreList1)],[mean(scoreList2)],[mean(scoreList3)]],["DPGAN","JSDGAN","SeqGAN"],"GANs","Promedio de puntaje Bleu-4","Comparación de GANS promediada")

#Grafico de promedio de puntajes
graphType2([scoreList1,scoreList2,scoreList3],["DPGAN","JSDGAN","SeqGAN"],"ID del Tweet generado","Puntaje Bleu-4","Comparación entre GANS")