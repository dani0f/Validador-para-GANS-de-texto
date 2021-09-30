
from nltk.translate.bleu_score import sentence_bleu
import csv

def BleuScore(text,referenceList):
	reference = []
	for sentence in referenceList:
		separated = sentence.split()
		reference.append(separated)
	candidate = text.split()
	return(sentence_bleu(reference, candidate))



referencesStringLen = 22  #numero de caracteres minimo de una referencia
referenceLen = 100        #numero de referencias utilizadas
referenceList = []        
with open('corpus_good.csv', newline='') as File:  
        reader = csv.reader(File)
        for row in reader:
                if(len(row[0]) >=  referencesStringLen):
                        referenceList.append(row[0])
                if(len(referenceList)==referenceLen):
                        break
print('BLEU score -> {}'.format(BleuScore("corona has become something else ugandan airport has misunderstood everything now extorting money from",referenceList)))
print('BLEU score -> {}'.format(BleuScore("corona has become something else flowers airport has misunderstood everything now extorting money from",referenceList)))	     
