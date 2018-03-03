'''
    Stephen Hung
    Implementation of Random Forest Classifier
    Trained using data from iris.data: https://archive.ics.uci.edu/ml/datasets/iris
'''

import csv
import sys
from random import seed
from random import randrange
from math import sqrt

STRING_TO_INT = {} # dictionary containing the string to integer values.

def load_data(filename):
    dataset = list()
    with open(filename, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            dataset.append(row)
    return dataset

def convert_dataset_strings(dataset):
    # Convert strings into float representation in the dataset.
    global STRING_TO_INT

    int_representation = 0
    for row in dataset:
        for index in range(0, len(row)):
            try:
                row[index] = float(row[index])
            except Exception:
                if(row[index] not in STRING_TO_INT):
                    STRING_TO_INT[row[index]] = int_representation
                    row[index] = STRING_TO_INT[row[index]]
                    int_representation+=1
                else:
                    row[index] = STRING_TO_INT[row[index]]
    return dataset

'''
    split decision trees on several split points based on attribute and value of attribute.
    Cost function is the Gini index

    find best split point: evaluate cost of each value in training dataset for each input variable

    sample of input attributes chosen randomly and without replacement - 
        each input variable considered once when looking for split point with lowest cost
'''
def get_split(dataset, num_features):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = sys.maxsize, sys.maxsize, sys.maxsize, None
    features = list()

    # get random list of indices for features.
    while len(features) < num_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)

    for index in features:
        for row in dataset:
            #groups = split the decision tree based on (index, row[index])
            #gini value = gini index based on (groups, class_values)
            #if gini value < best_score then replace all values best_score = gini

    return {'index': best_index, 'value':best_value, 'score':best_score, 'groups':best_groups}



def main():
    global NUM_STRING_COLUMN

    filename = 'iris.data'
    dataset = load_data('iris.data')
    dataset = convert_dataset_strings(dataset)

    dataset.remove([])

    total_input_feature = len(dataset[0]) - 1
    num_feature_split = float(sqrt(total_input_feature)) 
    # num of attributes is the bound for the number of features to be considered.
    
    get_split(dataset, num_feature_split)


if __name__ == '__main__':
    main()

 
'''
# Test the random forest algorithm
seed(2)
# load and prepare data
filename = 'iris.data'
dataset = load_data(filename)
print(dataset)
# convert string attributes to integers
str_column_to_float(dataset, 4)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [1, 5, 10]:
    scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
'''

