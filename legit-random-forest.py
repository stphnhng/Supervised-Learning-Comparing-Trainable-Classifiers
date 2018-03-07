'''
    Stephen Hung
    PseudoCode: http://proceedings.mlr.press/v28/denil13-supp.pdf
    https://onlinecourses.science.psu.edu/stat857/node/181

    Gini Impurity:
        https://datascience.stackexchange.com/questions/10228/gini-impurity-vs-entropy
        https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
'''
import csv
import sys
import random
from math import sqrt

STRING_TO_INT = {} # dictionary containing the string to integer values.

def load_data(filename): # loads data as a list.
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
                if(row[index] not in STRING_TO_INT): # increasing int representation for different items (0,1,2,...)
                    STRING_TO_INT[row[index]] = int_representation
                    row[index] = STRING_TO_INT[row[index]]
                    int_representation+=1
                else:
                    row[index] = STRING_TO_INT[row[index]]
    return dataset

'''
    Creates a random sample set from dataset with sample size of sample_size
    Chooses from dataset with replacement.
'''
def random_sample(dataset, sample_size):
    sample = list()
    while len(sample) < sample_size:
        # choose from data with replacement
        index = random.randrange(len(dataset))
        # without replacement:
        '''
        index = random.randrange(len(dataset))
        dataset.pop(index)
        '''
        sample.append(dataset[index])
    return sample



'''
    Calculate the Gini Impurity for a set of items (group)
        i is in {1,2,.... # of classes}
        p_i = fraction of items labeled with class i in data
        Gini Impurity = 1- sum(i = 1 to # of classes) p_i ^2
'''
def gini_impurity(group, classes):
    num_classes = len(classes)
    impurity_score = 0
    for i in classes:
        class_i_count = 0
        for row in group:
            if(row[-1] == i):
                class_i_count+=1
        # fraction of items with class i
        p_i = class_i_count * 1.0 / len(group) 
        impurity_score += (p_i)*(p_i)
    return 1 - impurity_score

def find_split_point(dataset, num_features):
    list_of_classes = set() # set because it disallows duplicates
    for row in dataset:
        list_of_classes.add(row[-1])
    list_of_classes = list(list_of_classes) # make it a list to make life easier

    # (2.) take random sample without replacement of the predictors

    # length of row without class at the end
    length_of_row = len(dataset[0]) - 1

    class_features = list() # random list of class features of size num_features
    while(len(class_features) < num_features):
        feature_index = random.randrange(length_of_row)
        if feature_index not in class_features:
            class_features.append(feature_index)

    '''
        find a split point?
    '''

    # (3.) Construct a split by using predictors selected in Step 2

    for feature_index in class_features:
        for row in dataset:


    







# n_samples = target values (class labels)
# n_features = features of a class

'''
    (1.) Take a random sample of size N with replacemenet from the data (bootstrap sample)

    (2.) Take a random sample without replacement of the predictors

    (3.) Construct a split by using predictors selected in Step 2

    Repeat 2 & 3 until tree is large

    Drop the out-of-bag data down the tree - store class assigned to each observation

    Repeat 1 - 5 a large amount of times

    for each observation, count the number of trees that is classified in one category

    assign each observation to a final category by a majority vote
'''



def main():
    dataset = load_data('iris.data')
    dataset = convert_dataset_strings(dataset)

    # clean data set so there are no empty rows
    dataset_length = len(dataset)
    for index in range(dataset_length):
        if not dataset[index]: # if empty, pop it
            dataset.pop(index)

    # (1.) get test data
    # Sample Size - Size of Test data to be used to identify accuracy of Classifier
    #             - Default is 50% of Total Dataset
    length_of_sample = 0.5 * len(dataset)
    random_sample(dataset, length_of_sample)    

    # 3 Main Tuning Parameters
    # Node Size - number of observations in terminal nodes of each tree of the forest
    #           - can be very small
    node_size = 1
    # Number of Trees - Number of Decision Trees
    #                 - 500 is default
    num_tree = 500
    # Number of Predictors Sampled - number of predictors sampled at each split
    #                              - Sampling 2 -5 each time is often adequate
    num_predictors = float(sqrt(len(dataset[0])-1))


    find_split_point(dataset, num_predictors)



if __name__ == '__main__':
    main()