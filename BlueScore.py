from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import csv

def csvToList(maxCorpusLen,minStringLen,filename):
        csvList = []        
        with open(filename, newline='') as File:  
                reader = csv.reader(File)
                for row in reader:
                        if(len(row[0]) >= minStringLen):
                                csvList.append(row[0].split())
                        if(len(csvList)==maxCorpusLen):
                                break
        return csvList


def BleuScore(referencesList,candidate):
        weights=(0.25,0.25,0.25,0.25)
        smoothing_function= SmoothingFunction().method4
        auto_reweigh = False
        return(sentence_bleu(referencesList, candidate, weights=weights,smoothing_function=smoothing_function,auto_reweigh=auto_reweigh))


def BleuScoreFromTxt(referencesList,minStringLen,filename):
    scoreList = []
    with open(filename, newline='') as File:  
                reader = File.read().split("\n")
                for row in reader:
                        if(len(row) >= minStringLen):
                                scoreList.append(BleuScore(referencesList,row.split()))
    return scoreList
    
    
