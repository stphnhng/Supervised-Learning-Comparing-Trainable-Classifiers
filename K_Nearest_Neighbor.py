import csv
import random
import math
from operator import itemgetter

#this function would first extract the data from the given files, and
#split the data based on the percentage given, into the training set and the test set
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
#calculate the euclidean distance from one data set to the other data set
#in order to determine how close the 2 datas are. This will be used to calculate
#the nearest neighbor of the given data set.
def euclideanDistance(data1, data2):
	sum = 0
	for i in range(len(data1)):
		try:
			sum = sum + math.sqrt(math.pow(data1[i] - data2[i],2))
		except:
			continue
	return sum

#calculate k data points that are nearest to the given starting point.This
#function would return a list of such nodes.
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
#given the k nearest neighbors, we would classify the given data. The way the
#algorithm works is that it would see the classification of the neighboring data points
#and it would classify the given data point based on the most common classification of its 
#neighbor. 
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
#this function would check how many data points in the test set are 
#correctly identified. It would return a percentage of accurate classification
def check(test, answer):
	count = 0
	for i in range(len(test)):
		if(answer[i] == test[i][-1]):
			count = count + 1	
	return float(count)/float(len(test)) * 100

#this is the main method that lets the users interact with the given program.
#this main method would allow the user to choose either the iris dataset or the
#ecoli dataset, and also it would also let the users choose the number of neighbors that
#they want to examine. Finally, this funtion would also show how long it would take 
#for the program to run.
def main():
	import os.path
	import time
	start = time.time()
	trainingSet = []
	testSet = []
	data_set = int(input("Choose 1 for Iris dataset and 2 for Ecoli dataset "))
	if(data_set == 1):
		setTrainTestData('iris.data', 0.70, trainingSet, testSet)
	else:
		setTrainTestData('ecoli.csv', 0.70, trainingSet, testSet)
	input_k = input("Enter k ")
	if(int(input_k) <= 0):
		input_k = 3
	predicted_value = []
	print("training complete")
	for i in range(len(testSet)):
		nearestNeighbor = findNearestNeighbor(int(input_k), trainingSet, testSet[i])
		predicted_value.append(classify(nearestNeighbor))
		print('> predicted=' + repr(classify(nearestNeighbor)) + ', actual=' + repr(testSet[i][-1]))
	print("Accuracy rate "+str(check(testSet, predicted_value)))
	end = time.time()
	print("This program runs in "+str(time.time() - start)+" seconds")


if __name__ == "__main__":
	main()

