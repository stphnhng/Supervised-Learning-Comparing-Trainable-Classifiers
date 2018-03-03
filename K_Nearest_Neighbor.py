import csv
import random
import math
from operator import itemgetter

def setTrainTestData(files, percentage, training = [], test = []):
	with open(files, 'rt') as csvfile:
		data = csv.reader(csvfile)
		dataList = list(data)
		for i in range(len(dataList) - 1):
			for j in range(len(dataList[0]) - 1):
				dataList[i][j] = float(dataList[i][j])
				if(random.random() < percentage):
					training.append(dataList[i])
				else:
					test.append(dataList[i])

def euclideanDistance(data1, data2):
	sum = 0
	for i in range(4):
		sum = sum + math.sqrt(math.pow(data1[i],2) + math.pow(data2[i],2))
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
	for i in nearestNodes:
		cur = i[-1]
		if cur in count:
			count[cur] = count[cur] + 1
		else:
			count[cur] = 1
	mostLikely = None
	mostLikelyNum = -1
	for i in count:
		if(count[i] > mostLikelyNum):
			mostLikely = i
			mostLikelyNum = count[i]
	return mostLikely
def check(test, answer):
	count = 0
	for i in range(len(test)):
		if(answer[i] == test[i][-1]):
			count = count + 1

	return float(count)/float(len(test)) * 100


def main():
	trainingSet = []
	testSet = []
	setTrainTestData('iris.data', 0.66, trainingSet, testSet)
	k = 2
	predicted_value = []
	for i in testSet:
		nearestNeighbor = findNearestNeighbor(k, trainingSet, trainingSet[0])
		predicted_value.append(classify(nearestNeighbor))
	print(str(check(testSet, predicted_value)))

if __name__ == "__main__":
	main()

