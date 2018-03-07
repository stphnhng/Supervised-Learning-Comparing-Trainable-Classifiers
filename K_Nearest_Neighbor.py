import csv
import random
import math
from operator import itemgetter

def setTrainTestData(files, percentage, training = [], test = []):
	with open(files, 'rt') as csvfile:
		data = csv.reader(csvfile)
		dataList = list(data)
		for i in range(len(dataList) - 1):
			for j in range(len(dataList[0])):
				try:
					dataList[i][j] = float(dataList[i][j])
				except ValueError:
					continue
			if(random.random() < percentage):
				training.append(dataList[i])
			else:
				test.append(dataList[i])
def euclideanDistance(data1, data2):
	sum = 0
	for i in range(len(data1)):
		try:
			sum = sum + math.sqrt(math.pow(data1[i] - data2[i],2))
		except:
			continue
	return sum

def findNearestNeighbor(k, train, startingPoint):
	nearestNodes = []
	distancesAll = []
	for i in range(len(train)):
		distance_cur = euclideanDistance(train[i], startingPoint)
		distancesAll.append((distance_cur, train[i]))
	distancesAll.sort(key = itemgetter(0))
	for i in range(k):
		nearestNodes.append(distancesAll[i][1])
	return nearestNodes
def classify(nearestNodes):
	count = {}
	#print(nearestNodes)
	for i in range(len(nearestNodes)):
		cur = nearestNodes[i][-1]
		if cur in count:
			count[cur] = count[cur] + 1
		else:
			count[cur] = 1
	mostLikely = None
	mostLikelyNum = -1
	for i in count:
		if(count[i] > mostLikelyNum):
			#print(i)
			mostLikely = i
			mostLikelyNum = count[i]

	#print(mostLikely)
	return mostLikely
def check(test, answer):
	count = 0
	for i in range(len(test)):
		if(answer[i] == test[i][-1]):
			count = count + 1	
	return float(count)/float(len(test)) * 100


def main():
	import os.path
	import time
	start = time.time()
	trainingSet = []
	testSet = []
	#dataset = raw_input("Which edataset should the program use?")
	#if(os.path.exists(dataset)):
	#	setTrainTestData('dataset', 0.66, trainingSet, testSet)
	#else:
	data_set = int(input("Choose 1 for Iris dataset and 2 for Ecoli dataset "))
	if(data_set == 1):
		setTrainTestData('iris.data', 0.70, trainingSet, testSet)
	else:
		setTrainTestData('ecoli.csv', 0.70, trainingSet, testSet)
	input_k = input("Enter k ")
	if(int(input_k) <= 0):
		input_k = 3
	predicted_value = []
	for i in range(len(testSet)):
		nearestNeighbor = findNearestNeighbor(int(input_k), trainingSet, testSet[i])
		predicted_value.append(classify(nearestNeighbor))
		print('> predicted=' + repr(classify(nearestNeighbor)) + ', actual=' + repr(testSet[i][-1]))
	print("Accuracy rate "+str(check(testSet, predicted_value)))
	end = time.time()
	print("This program runs in "+str(time.time() - start)+" seconds")


if __name__ == "__main__":
	main()

