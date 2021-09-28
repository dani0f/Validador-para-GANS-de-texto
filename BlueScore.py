
from nltk.translate.bleu_score import sentence_bleu

def BleuScore(text,referenceList):
	print(referenceList)
	reference = []
	for sentence in referenceList:
		separated = sentence.split()
		print(separated)
		reference.append(separated)
	print(reference)
	candidate = text.split()
	print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))


BleuScore("First test of fake data",["First test of real data","Second test of real data"])
BleuScore("First test of real data",["First test of real data","Second test of real data"])